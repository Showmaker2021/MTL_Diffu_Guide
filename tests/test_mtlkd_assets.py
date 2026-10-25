"""Check teacher acquisition without using the network or real weights."""

from io import BytesIO
import hashlib

import pytest

from baselines.mtlkd import assets


def test_download_is_pinned_verified_and_reusable(tmp_path, monkeypatch):
    content = b'test teacher checkpoint'
    manifest = dict(revision='fixed-revision', files={
        'cvrp100.pt': dict(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())})
    monkeypatch.setattr(assets, 'MANIFEST', manifest)
    calls = []

    def download(url, timeout):
        calls.append(url)
        return BytesIO(content)

    monkeypatch.setattr(assets, 'urlopen', download)
    paths = assets.ensure_teachers(['CVRP'], tmp_path)
    assert paths[0].read_bytes() == content
    assert '/fixed-revision/pretrained/cvrp100.pt' in calls[0]
    assert assets.ensure_teachers(['CVRP'], tmp_path) == paths
    assert len(calls) == 1
    assert not list(tmp_path.glob('*.part'))
    paths[0].write_bytes(b'corrupted cache')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        assets.ensure_teachers(['CVRP'], tmp_path)


def test_invalid_download_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'urlopen', lambda *args, **kwargs: BytesIO(b'invalid'))
    with pytest.raises(ValueError, match='integrity check'):
        assets.ensure_teachers(['CVRP'], tmp_path)
    assert not list(tmp_path.iterdir())
