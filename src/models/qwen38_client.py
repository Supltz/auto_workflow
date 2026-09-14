"""Qwen3.8 client with a vLLM OpenAI endpoint and Transformers fallback."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import Any, TypeVar

import requests
from PIL import Image
from pydantic import BaseModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

_POOL_LOCK = threading.Lock()


@contextmanager
def _allocation_lock(root: Path):
    """Serialize ready-service selection with a GPU owner's drain transition."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "allocation.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _metric_load(endpoint: str) -> int:
    base = endpoint.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        response = requests.get(base + "/metrics", timeout=0.5)
        response.raise_for_status()
    except requests.RequestException:
        return 1 << 20
    total, found = 0, False
    for metric in ("vllm:num_requests_running", "vllm:num_requests_waiting",
                   "vllm_num_requests_running", "vllm_num_requests_waiting"):
        for match in re.finditer(
            r"^" + re.escape(metric) + r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)",
            response.text, flags=re.MULTILINE,
        ):
            total += max(0, int(float(match.group(1))))
            found = True
    return total if found else 0


def _reserve_once(model: str, fallback: str, timeout: float, excluded: set[str]):
    """Choose the least-loaded ready endpoint and publish a short local lease."""
    root_value = os.environ.get("QWEN_NODE_POOL")
    if not root_value:
        return fallback, None
    root = Path(root_value)
    # The filesystem lock is shared with qwen_pool.Service.drain().  Once drain
    # publishes "draining", no client can create a new lease for that GPU.
    with _POOL_LOCK, _allocation_lock(root):
        candidates = []
        now = time.time()
        for path in (root / "services").glob("*.json"):
            try:
                value = json.loads(path.read_text())
                endpoint = str(value["endpoint"]).rstrip("/")
                if (value.get("version") != 1 or value.get("status") != "ready"
                        or value.get("model") != model or endpoint in excluded
                        or not _pid_alive(int(value["pid"]))):
                    continue
                directory = root / "reservations" / path.stem
                directory.mkdir(parents=True, exist_ok=True)
                active = 0
                for ticket in directory.glob("*.json"):
                    try:
                        ticket_value = json.loads(ticket.read_text())
                        expired = float(ticket_value.get("expires", 0)) < now
                        dead = not _pid_alive(int(ticket_value.get("pid", -1)))
                        if expired or dead:
                            ticket.unlink(missing_ok=True)
                        else:
                            active += 1
                    except (OSError, ValueError, TypeError, AttributeError):
                        ticket.unlink(missing_ok=True)
                candidates.append((endpoint, directory, active))
            except (OSError, ValueError, TypeError, KeyError):
                continue
        if not candidates:
            if os.environ.get('OPD_RESIDENT_POOL'):
                # Never bypass the per-GPU gate by using a draining fallback.
                return None, None
            return fallback, None
        scored = [(_metric_load(endpoint) + active, endpoint, directory)
                  for endpoint, directory, active in candidates]
        _, endpoint, directory = min(scored, key=lambda item: (item[0], item[1]))
        ticket = directory / (uuid.uuid4().hex + ".json")
        ticket.write_text(json.dumps({"pid": os.getpid(), "expires": now + timeout + 60}))
    return endpoint, ticket


def _reserve_endpoint(model: str, fallback: str, timeout: float, excluded: set[str]):
    deadline = time.monotonic() + timeout
    while True:
        endpoint, ticket = _reserve_once(model, fallback, timeout, excluded)
        if endpoint is not None:
            return endpoint, ticket
        if time.monotonic() >= deadline:
            raise requests.Timeout('No ready Qwen endpoint; waiting for GPU inference slot timed out')
        time.sleep(0.1)


class QwenValidationError(ValueError):
    """Model output remained invalid after bounded correction attempts."""


class QwenContextLengthError(QwenValidationError):
    """The server explicitly rejected an over-context request; skip without retry."""
    reject_reason = "context_length_exceeded"


class ModelContractError(ValueError):
    """An explicitly checked model answer violates the request contract."""


def _data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response contains no JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("model response JSON root is not an object")
    return value


def _parse_for_schema(text: str, schema: type[BaseModel]) -> dict[str, Any]:
    """Parse the one strict JSON object returned by a Route B Qwen stage."""
    del schema  # The caller performs schema validation immediately after parsing.
    return parse_json_object(text)


class Qwen38Client:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.backend = config.get("backend", "vllm")
        self._model: Any = None
        self._processor: Any = None

    def generate_json(
        self,
        prompt: str,
        images: list[Image.Image],
        schema: type[SchemaT],
        extra_text: str | None = None,
        validate_result: Callable[[SchemaT], None] | None = None,
        expected_fields: dict[str, str] | None = None,
    ) -> tuple[SchemaT, str]:
        full_prompt = prompt if not extra_text else f"{prompt}\n\n{extra_text}"
        if expected_fields:
            full_prompt += "\nCopy these authoritative output fields exactly: " + json.dumps(expected_fields)
        last_error: Exception | None = None
        for attempt in range(int(self.config.get("max_attempts", 3))):
            attempt_prompt = full_prompt
            if attempt:
                attempt_prompt += (
                    "\n\nYour prior response did not validate. Return only one JSON object matching "
                    "the requested schema, with no markdown or commentary."
                    f"\nValidation error to correct: {str(last_error)[:2000]}"
                )
            # Service/protocol/runtime errors are not model-answer failures.
            if self.backend == "vllm":
                raw = self._vllm_generate(attempt_prompt, images, schema)
            elif self.backend == "transformers":
                raw = self._transformers_generate(attempt_prompt, images)
            else:
                raise ValueError(f"unsupported Qwen backend: {self.backend}")
            if not isinstance(raw, str):
                raise TypeError("model endpoint returned non-text content")
            try:
                result = schema.model_validate(_parse_for_schema(raw, schema))
            except (ValueError, TypeError) as exc:
                last_error = exc
                continue
            try:
                for key, value in (expected_fields or {}).items():
                    if getattr(result, key) != value:
                        raise ModelContractError(f"{key} must be exactly {value!r}")
                if validate_result is not None:
                    validate_result(result)
                return result, raw
            except ModelContractError as exc:
                last_error = exc
        raise QwenValidationError(f"Qwen failed JSON schema validation: {last_error}") from last_error

    def _content(self, prompt: str, images: list[Image.Image]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": _data_url(image)}} for image in images
        ]
        content.append({"type": "text", "text": prompt})
        return content

    def _vllm_generate(
        self,
        prompt: str,
        images: list[Image.Image],
        schema: type[BaseModel],
    ) -> str:
        fallback = os.environ.get("QWEN_TRANSPORT_API_BASE", self.config["api_base"]).rstrip("/")
        payload = {
            "model": self.config["name"],
            "messages": [{"role": "user", "content": self._content(prompt, images)}],
            "temperature": self.config.get("temperature", 0.0),
            "max_tokens": self.config.get("max_output_tokens", 2048),
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            },
        }
        headers = {"Authorization": f"Bearer {self.config.get('api_key', 'local')}"}
        # Transport retries are separate from model-answer/schema correction.
        # Resend the full multimodal payload; never silently discard a target on 5xx.
        retry_delays = (2, 5)
        excluded: set[str] = set()
        response = None
        transport_error: requests.RequestException | None = None
        request_timeout = float(self.config.get("request_timeout_seconds", 600))
        for attempt in range(len(retry_delays) + 1):
            endpoint, ticket = _reserve_endpoint(
                self.config["name"], fallback, request_timeout, excluded,
            )
            try:
                response = requests.post(
                    endpoint + "/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=request_timeout,
                )
                transport_error = None
            except requests.RequestException as exc:
                transport_error = exc
                excluded.add(endpoint)
                delay = retry_delays[attempt] if attempt < len(retry_delays) else None
                print(
                    f"[qwen_http] endpoint={endpoint} attempt={attempt + 1}/3 "
                    f"transport_error={str(exc)[:2000]!r} "
                    + (f"retry_in={delay}s" if delay is not None else "retries_exhausted"),
                    flush=True,
                )
                if delay is None:
                    raise
                time.sleep(delay)
                continue
            finally:
                if ticket is not None:
                    ticket.unlink(missing_ok=True)
            if response.status_code not in {500, 502, 503, 504}:
                break
            excluded.add(endpoint)
            try:
                body = response.json()
            except ValueError:
                body = {}
            error = body.get("error", body) if isinstance(body, dict) else {}
            detail = error.get("message") if isinstance(error, dict) else None
            delay = retry_delays[attempt] if attempt < len(retry_delays) else None
            print(
                f"[qwen_http] status={response.status_code} attempt={attempt + 1}/3 "
                f"server_error={str(detail or response.text)[:2000]!r} "
                + (f"retry_in={delay}s" if delay is not None else "retries_exhausted"),
                flush=True,
            )
            if delay is None:
                break  # Existing HTTPError handling below keeps exhaustion fatal.
            response.close()
            time.sleep(delay)
        if response is None:
            assert transport_error is not None
            raise transport_error
        if response.status_code >= 400:
            # Inspect only the server error, never the submitted image/prompt payload.
            try:
                body = response.json()
            except ValueError:
                body = {}
            error = body.get("error", body) if isinstance(body, dict) else {}
            message = str(error.get("message", "")) if isinstance(error, dict) else ""
            code = error.get("code") if isinstance(error, dict) else None
            context_overflow = (
                code == "context_length_exceeded"
                or bool(re.search(
                    r"input length .*exceeds .*maximum context length"
                    r"|(?:prompt|input).*longer than the maximum (?:model|context) length"
                    r"|maximum context length.*(?:requested|resulted|exceed)",
                    message, flags=re.IGNORECASE | re.DOTALL,
                ))
            )
            if response.status_code == 400 and context_overflow:
                raise QwenContextLengthError(message or "context_length_exceeded")
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = message or response.text[:2000]
                raise requests.HTTPError(
                    f"{exc}; server_error={detail[:2000]}",
                    response=response, request=exc.request,
                ) from exc
        return response.json()["choices"][0]["message"]["content"]

    def _load_transformers(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        model_path = str(Path(self.config["local_path"]).resolve())
        dtype = torch.bfloat16 if self.config.get("dtype") == "bfloat16" else torch.float16
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    def _transformers_generate(self, prompt: str, images: list[Image.Image]) -> str:
        self._load_transformers()
        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": image} for image in images),
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        generated = self._model.generate(
            **inputs,
            max_new_tokens=int(self.config.get("max_output_tokens", 2048)),
            do_sample=False,
        )
        trimmed = generated[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(trimmed, skip_special_tokens=True)[0]
