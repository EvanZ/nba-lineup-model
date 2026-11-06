import shutil
from pathlib import Path
import pytest
from test_environment_selection import FakeS3
from nba_lineup_model.players.reconciliation_artifact import accept, reference, seal, selected, verify

LAKE = 's3://nba-gestalt-data/lake'


def bundle(root, *, parent=None, label='first'):
    root.mkdir()
    for name in ('rankings.parquet','profiles.parquet','minutes.json'):
        (root/name).write_bytes(label.encode())
    seal(root, {'season':'2026-27', 'effective_through':'2026-11-04',
        'parent_reference': parent, 'introductions': [], 'dependencies': []})
    return root


def publish(root, tmp, client, **kwargs):
    return accept(root, environment='dev', state_root=tmp/'state', cache_root=tmp/'cache',
                  lake_uri=LAKE, client=client, **kwargs)


def test_recovery_uses_s3_after_all_local_reconciliation_state_is_deleted(tmp_path):
    client=FakeS3(); root=bundle(tmp_path/'build')
    result=publish(root,tmp_path,client)
    shutil.rmtree(root);shutil.rmtree(tmp_path/'state');shutil.rmtree(tmp_path/'cache',ignore_errors=True)
    recovered=selected(environment='dev',cache_root=tmp_path/'empty',lake_uri=LAKE,client=client)
    assert reference(verify(recovered)).as_dict()==result['reference']
    assert (recovered/'rankings.parquet').read_bytes()==b'first'


def test_partial_upload_does_not_select_and_identical_retry_completes(tmp_path,monkeypatch):
    client=FakeS3();root=bundle(tmp_path/'build');original=client.put_object
    def interrupted(**kw):
        if kw['Key'].endswith('profiles.parquet'):raise RuntimeError('interrupted upload')
        return original(**kw)
    monkeypatch.setattr(client,'put_object',interrupted)
    with pytest.raises(RuntimeError):publish(root,tmp_path,client)
    assert selected(environment='dev',cache_root=tmp_path/'cache',lake_uri=LAKE,client=client) is None
    assert not (tmp_path/'state/selection.json').exists()
    monkeypatch.setattr(client,'put_object',original)
    first=publish(root,tmp_path,client);second=publish(root,tmp_path,client)
    assert first==second
    assert first['generation']==1


def test_successive_versions_and_stale_writer(tmp_path):
    client=FakeS3();first=bundle(tmp_path/'one');r=publish(first,tmp_path,client)
    second=bundle(tmp_path/'two',parent=r['reference'],label='second')
    stale=bundle(tmp_path/'stale',parent=r['reference'],label='stale')
    assert publish(second,tmp_path,client)['generation']==2
    with pytest.raises(ValueError,match='stale'):publish(stale,tmp_path,client)
    with pytest.raises(ValueError,match='stale'):publish(first,tmp_path,client)


def test_corrupt_catalog_never_publishes(tmp_path):
    client=FakeS3();root=bundle(tmp_path/'build');(root/'minutes.json').write_text('changed')
    with pytest.raises(ValueError,match='verification'):publish(root,tmp_path,client)
    assert not client.objects
