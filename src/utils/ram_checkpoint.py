"""Cooperative, immutable RAM snapshots. Only complete published generations resume.

Called at a serialized request boundary, never by a background thread racing the
writer. Runtime settings are supplied externally; disabled for ordinary runs.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time
import uuid
import stat


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temp.open('w') as f:
            json.dump(value, f, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        sync_dir(path.parent)
    finally: temp.unlink(missing_ok=True)


def files(root):
    result = {}
    def visit(directory,prefix=''):
        with os.scandir(directory) as entries:
            for entry in entries:
                if (not prefix and entry.name=='logs') or entry.name.startswith('.ram-'):continue
                relative=prefix+entry.name
                if entry.is_symlink():raise ValueError(f'Unexpected task symlink: {entry.path}')
                if entry.is_dir(follow_symlinks=False):visit(entry.path,relative+'/')
                else:
                    s=entry.stat(follow_symlinks=False)
                    if stat.S_ISREG(s.st_mode):result[relative]=(s.st_size,s.st_mtime_ns)
    visit(root)
    return result


def safe_member(name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or name == '.':
        raise ValueError('Unsafe archive member: ' + name)
    return p


def commit(root, store, identity, *, max_bytes=32 << 30):
    """Caller owns task lock and has stopped mutation at a request boundary."""
    root, store = Path(root), Path(store)
    store.mkdir(parents=True, exist_ok=True)
    state_path = root / '.ram-snapshot.json'
    prior = json.loads(state_path.read_text()) if state_path.exists() else {}
    latest = json.loads((store/'latest.json').read_text()) if (store/'latest.json').exists() else None
    if latest and latest['identity'] != identity: raise ValueError('Snapshot identity mismatch')
    if prior and (not latest or prior['generation'] != latest['generation']):
        prior = {}  # New owner must restore published data before running.
    current = files(root)
    if sum(v[0] for v in current.values()) > max_bytes: raise RuntimeError('RAM task budget exceeded')
    old = prior.get('files', {})
    changed = [n for n,s in current.items() if list(s) != old.get(n)]
    removed = sorted(set(old)-set(current))
    if prior and not changed and not removed: return latest
    generation = uuid.uuid4().hex
    archive = root / ('.ram-' + generation + '.tar')
    remote_temp = store / ('.' + generation + '.tmp')
    try:
        with tarfile.open(archive, 'w') as tar:
            for name in changed: tar.add(root/name, arcname=name, recursive=False)
        # No checkpoint can claim a concurrently modified file.
        if files(root) != current: raise RuntimeError('Task changed during snapshot')
        checksum = digest(archive)
        with archive.open('rb') as src, remote_temp.open('xb') as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
            dst.flush(); os.fsync(dst.fileno())
        if digest(remote_temp) != checksum: raise IOError('Checkpoint transfer verification failed')
        bundle = generation + '.tar'
        os.replace(remote_temp, store/bundle); sync_dir(store)
        chain = list(latest['chain']) if prior else []
        chain.append(dict(bundle=bundle, sha256=checksum, removed=removed))
        receipt = dict(version=1, identity=identity, generation=generation, chain=chain,
                       files=sorted(current), committed_at=time.time())
        # This is the sole durable commit point. Orphan bundles are ignored.
        atomic(store/'latest.json', receipt)
        atomic(state_path, dict(generation=generation, files=current))
        return receipt
    finally:
        archive.unlink(missing_ok=True)
        remote_temp.unlink(missing_ok=True)


def restore(store, destination, identity):
    store, destination = Path(store), Path(destination)
    receipt = json.loads((store/'latest.json').read_text())
    if receipt['identity'] != identity: raise ValueError('Snapshot identity mismatch')
    if destination.exists() and any(destination.iterdir()): raise ValueError('Restore requires empty directory')
    destination.mkdir(parents=True, exist_ok=True)
    for item in receipt['chain']:
        name = safe_member(item['bundle'])
        if len(name.parts) != 1: raise ValueError('Invalid bundle name')
        archive = store/name
        if digest(archive) != item['sha256']: raise IOError('Corrupt checkpoint bundle')
        with tarfile.open(archive) as tar:
            for member in tar:
                relative = safe_member(member.name)
                if not member.isfile(): raise ValueError('Only regular checkpoint files allowed')
                target = destination/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
        for name in item['removed']:
            (destination/safe_member(name)).unlink(missing_ok=True)
    actual = files(destination)
    if sorted(actual) != receipt['files']: raise IOError('Checkpoint file set mismatch')
    atomic(destination/'.ram-snapshot.json', dict(generation=receipt['generation'], files=actual))
    return receipt


_count = 0
_last = time.monotonic()

def checkpoint_boundary(*, force=False):
    """Synchronous safe point, at most configured requests between commits."""
    global _count, _last
    config = os.environ.get('OPD_RAM_CHECKPOINT')
    if not config: return
    _count += 1
    cfg = json.loads(config)
    if not force and _count < cfg.get('requests',32) and time.monotonic()-_last < cfg.get('seconds',60): return
    start = time.monotonic()
    receipt = commit(cfg['root'], cfg['store'], cfg['identity'], max_bytes=cfg.get('max_bytes',32<<30))
    print(f'[ram commit] generation={receipt["generation"]} elapsed={time.monotonic()-start:.2f}s',flush=True)
    _count=0; _last=time.monotonic()
