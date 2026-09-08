import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.models.qwen38_client import QwenValidationError
from src.routes.proposal_coverage import report_proposal_coverage
from src.routes.proposal_priority import prioritize_proposals
from src.routes.route_b_stages import extract_caption_entities
from src.schema import VisualEntitySet


def proposal(name, rank, span="", supported=False):
    return {
        "entity_id": name, "rank": rank, "caption_span": span,
        "caption_supported": supported, "source": "visual_proposal",
        "short_name": name, "proposal_reason": "clear bounded object",
        "category": "bag", "scope": "whole_object", "attributes": [name],
        "visible_text": [], "action": None, "relations": [],
        "category_query": "bag", "locator_query": "the " + name + " bag",
    }


class ProposalPriorityTests(unittest.TestCase):
    caption = "A red bag and a blue bag."

    def select(self, entities, cap=30):
        return prioritize_proposals(
            entities, caption=self.caption, image_id="sample", proposal_cap=cap)

    def test_caption_priority_before_cap_and_stable_order(self):
        entities = [proposal("yellow", 1), proposal("red", 2, "red bag", True),
                    proposal("blue", 3, "blue bag", True), proposal("green", 4)]
        original = copy.deepcopy(entities)
        result = self.select(entities, 3)
        self.assertEqual([e["short_name"] for e in result], ["red", "blue", "yellow"])
        self.assertEqual([e["rank"] for e in result], [1, 2, 3])
        self.assertEqual(entities, original)
        VisualEntitySet.model_validate({"entities": result})

    def test_invalid_caption_claim_does_not_take_priority(self):
        entities = [proposal("green", 1, "green bag", True),
                    proposal("red", 2, "red bag", True)]
        self.assertEqual(self.select(entities, 1)[0]["short_name"], "red")
        invalid = self.select(entities)[1]
        self.assertFalse(invalid["caption_supported"])
        self.assertEqual(invalid["caption_span"], "")

    def test_deduplication_does_not_waste_capacity(self):
        first = proposal("red", 1, "red bag", True)
        duplicate = {**first, "entity_id": "another-id", "rank": 2}
        result = self.select([first, duplicate, proposal("yellow", 3)], 2)
        self.assertEqual([e["short_name"] for e in result], ["red", "yellow"])

    def test_caption_targets_fill_cap_without_visual_fill(self):
        entities = [proposal("yellow", 1), proposal("red", 2, "red bag", True),
                    proposal("blue", 3, "blue bag", True)]
        self.assertEqual([e["short_name"] for e in self.select(entities, 1)], ["red"])

    def test_visual_only_empty_and_zero_cap(self):
        entities = [proposal("yellow", 1), proposal("green", 2)]
        self.assertEqual([e["short_name"] for e in self.select(entities)], ["yellow", "green"])
        self.assertEqual(self.select([]), [])
        self.assertEqual(self.select(entities, 0), [])

    def test_id_is_independent_of_rank_and_cap(self):
        entity = proposal("red", 1, "red bag", True)
        first = self.select([entity])[0]["entity_id"]
        second = self.select([proposal("yellow", 1), {**entity, "rank": 2}], 1)[0]["entity_id"]
        self.assertEqual(first, second)

    def test_one_request_existing_schema_and_prompt_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "image.png"
            Image.new("RGB", (64, 64)).save(image)
            prompt = root / "prompt.txt"
            prompt.write_text("caption-first contract")
            output = root / "entities.jsonl"
            result = VisualEntitySet.model_validate({"entities": [
                proposal("yellow", 1), proposal("red", 2, "red bag", True),
                proposal("blue", 3, "blue bag", True)]})
            kwargs = {
                "manifest_rows": [{"image_id": "sample", "image_path": str(image),
                                   "caption": self.caption, "width": 64, "height": 64}],
                "output_path": output, "failures_path": root / "failures.jsonl",
                "prompt_path": prompt, "qwen_config": {}, "config": {
                    "route": "B_caption", "max_entities_per_image": 2},
                "workers": 1, "resume": True, "overwrite": False,
            }
            with patch("src.routes.route_b_stages.model_metadata", return_value={}), patch(
                "src.routes.route_b_stages.Qwen38Client"
            ) as client_type:
                client = client_type.return_value
                client.generate_json.return_value = (result, "{}")
                extract_caption_entities(**kwargs)
                self.assertEqual(client.generate_json.call_count, 1)
                args, call = client.generate_json.call_args
                self.assertEqual(len(args[1]), 1)
                self.assertIs(args[2], VisualEntitySet)
                card = json.loads(call["extra_text"])
                self.assertEqual(card["proposal_policy"], "caption_first_then_visual_fill")
                self.assertEqual(card["caption_reference"], self.caption)
                row = json.loads(output.read_text())
                self.assertEqual([e["short_name"] for e in row["entities"]], ["red", "blue"])
                self.assertEqual(set(row["entities"][0]), set(result.model_dump()["entities"][0]))
                extract_caption_entities(**kwargs)
                self.assertEqual(client.generate_json.call_count, 1)
                prompt.write_text("changed caption-first contract")
                extract_caption_entities(**kwargs)
                self.assertEqual(client.generate_json.call_count, 2)
                self.assertTrue(list((root / "checkpoint_archive").glob("*.jsonl")))

    def test_empty_recheck_recovery_empty_failure_and_zero_cap(self):
        empty = VisualEntitySet(entities=[])
        recovered = VisualEntitySet.model_validate({
            "entities": [proposal("red", 1, "red bag", True)]})
        cases = [
            ("recovered", 2, [(empty, '{"entities":[]}'), (recovered, "recovered")], 1, False),
            ("still_empty", 2, [(empty, '{"entities":[]}'), (empty, '{"entities":[]}')], 0, False),
            ("failed_recheck", 2, [(empty, '{"entities":[]}'),
                                  QwenValidationError("invalid recheck")], 0, True),
            ("zero_cap", 0, [(empty, '{"entities":[]}')], 0, False),
        ]
        for name, cap, responses, count, invalid in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                image = root / "image.png"
                Image.new("RGB", (64, 64)).save(image)
                prompt = root / "prompt.txt"
                prompt.write_text("caption-first recall")
                output = root / "entities.jsonl"
                kwargs = {
                    "manifest_rows": [{"image_id": "sample", "image_path": str(image),
                                       "caption": self.caption, "width": 64, "height": 64}],
                    "output_path": output, "failures_path": root / "failures.jsonl",
                    "prompt_path": prompt, "qwen_config": {}, "config": {
                        "route": "B_caption", "max_entities_per_image": cap},
                    "workers": 1, "resume": True, "overwrite": False,
                }
                with patch("src.routes.route_b_stages.model_metadata", return_value={}), patch(
                    "src.routes.route_b_stages.Qwen38Client"
                ) as client_type:
                    client = client_type.return_value
                    client.generate_json.side_effect = responses
                    extract_caption_entities(**kwargs)
                    self.assertEqual(client.generate_json.call_count, len(responses))
                    row = json.loads(output.read_text())
                    self.assertEqual(len(row["entities"]), count)
                    self.assertEqual(bool(row.get("model_output_invalid")), invalid)
                    report = json.loads(output.with_suffix(".coverage.json").read_text())
                    self.assertEqual(report["invalid_images"], int(invalid))
                    self.assertEqual(report["empty_images"], int(not count and not invalid))
                    if not invalid:
                        self.assertEqual(row["metadata"]["empty_recheck"], cap > 0)
                        self.assertEqual(report["empty_rechecks_recovered"], int(count > 0))
                        if cap > 0:
                            calls = client.generate_json.call_args_list
                            self.assertIs(calls[0].args[1][0], calls[1].args[1][0])
                            card = json.loads(calls[1].kwargs["extra_text"])
                            self.assertTrue(card["empty_result_recheck"])
                            self.assertEqual(card["caption_reference"], self.caption)
                    extract_caption_entities(**kwargs)
                    self.assertEqual(client.generate_json.call_count, len(responses))

    def test_coverage_warns_for_valid_empty_but_separates_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "entities.jsonl"
            records = [{"entities": [], "metadata": {"empty_recheck": True}}
                       for _ in range(6)]
            records += [{"entities": [{}], "metadata": {"empty_recheck": True}}
                        for _ in range(4)]
            records += [{"entities": [], "model_output_invalid": True}]
            report = report_proposal_coverage(records, path)
            self.assertTrue(report["warning"])
            self.assertEqual(report["valid_images"], 10)
            self.assertEqual(report["empty_images"], 6)
            self.assertEqual(report["invalid_images"], 1)
            self.assertEqual(report["empty_rechecks_recovered"], 4)
            self.assertEqual(report["proposal_count_distribution"], {0: 6, 1: 4})
            only_errors = report_proposal_coverage(
                [{"entities": [], "model_output_invalid": True}], path)
            self.assertIsNone(only_errors["empty_fraction_of_valid_images"])
            self.assertEqual(only_errors["empty_images"], 0)


if __name__ == "__main__":
    unittest.main()
