import hashlib
import io
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from nba_lineup_model.storage.operational_reports import publish, read


class Store:
    def __init__(self):
        self.objects = {}
    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if kwargs.get('IfNoneMatch') == '*' and Key in self.objects:
            raise ClientError({'Error':{'Code':'PreconditionFailed'}}, 'PutObject')
        self.objects[Key] = Body
    def get_object(self, *, Bucket, Key, **kwargs):
        raw = self.objects[Key]
        return {'Body':io.BytesIO(raw),'ContentLength':len(raw),'ETag':hashlib.md5(raw).hexdigest()}


def test_report_snapshot_reads_s3_without_disk_and_is_idempotent(monkeypatch):
    store = Store()
    files = {'report.html':'<main>Actual findings</main>','alert-state.json':'{"active":{}}'}
    uri = 's3://test/reports/dev'
    location = publish(uri, 'data-quality', files, client=store)
    assert publish(uri, 'data-quality', files, client=store) == location
    monkeypatch.setattr(Path, 'open', lambda *a,**k: pytest.fail('local file access'))
    assert read(uri, 'data-quality', client=store) == files
    assert len(store.objects) == 2


def test_corrupt_or_missing_report_does_not_use_disk():
    from nba_lineup_model.season.durable_observations import _BYTES
    store = Store()
    uri = 's3://test/reports/dev'
    location = publish(uri, 'daily-pipeline', {'report.html':'actual run'}, client=store)
    key = location[5:].partition('/')[2]
    store.objects[key] = b'corrupt'
    _BYTES.clear()
    with pytest.raises(ValueError, match='mismatch'):
        read(uri, 'daily-pipeline', client=store)
    del store.objects[key]
    _BYTES.clear()
    with pytest.raises(ValueError, match='Cannot read required S3 object'):
        read(uri, 'daily-pipeline', client=store)


def test_website_registry_writes_are_conditional_and_never_touch_disk(monkeypatch):
    from nba_lineup_model.storage.website_data import ServingData
    store = Store()
    reader = ServingData(Path('/absent/release'), lake_uri='s3://test/lake', client=store)
    reader.registries['/absent/registry'] = {'source_root':'/absent/registry','uri':'s3://test/registry','records':{}}
    def head(*, Bucket, Key):
        raw=store.objects[Key]
        return {'ContentLength':len(raw),'Metadata':{'sha256':hashlib.sha256(raw).hexdigest()}}
    store.head_object = head
    monkeypatch.setattr(Path, 'open', lambda *a,**k: pytest.fail('local registry access'))
    path = Path('/absent/registry/selections/example.json')
    reader.put_registry(path, b'{"forecast":"immutable"}\n')
    reader.put_registry(path, b'{"forecast":"immutable"}\n')
    assert reader.read(path) == b'{"forecast":"immutable"}\n'
    with pytest.raises(ValueError, match='Conflicting immutable S3 registry publication'):
        reader.put_registry(path, b'{"forecast":"replacement"}\n')
    assert len(store.objects) == 1


def test_both_daily_report_producers_publish_their_exact_status(tmp_path, monkeypatch):
    import json
    from nba_lineup_model.storage import operational_reports
    from nba_lineup_model.orchestration.daily_pipeline import _report
    from nba_lineup_model.orchestration.daily_work import write_work_report
    calls=[]
    monkeypatch.setattr(operational_reports, 'publish', lambda uri,kind,files: calls.append((uri,kind,files)))
    status={'game_date':'2026-11-04','status':'failed','stage':'validation','error':'Actual failure'}
    _report(tmp_path, status, s3_uri='s3://test/reports/dev')
    assert json.loads(calls[-1][2]['report.json'])==status
    assert calls[-1][2]['report.html']==(tmp_path/'report.html').read_text()
    write_work_report(tmp_path, {'work_id':'immutable-work'}, status, [], s3_uri='s3://test/reports/dev')
    assert len(calls)==2
    assert json.loads(calls[-1][2]['report.json'])==status
    assert calls[-1][2]['report.html']==(tmp_path/'report.html').read_text()
    assert all(c[:2]==('s3://test/reports/dev','daily-pipeline') for c in calls)
