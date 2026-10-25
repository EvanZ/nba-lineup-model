from __future__ import annotations

import hashlib
from io import BytesIO

from botocore.exceptions import ClientError

from nba_lineup_model.storage.schedule_injury_archive import backfill_legacy_injury_pdfs


class FakeS3:
    def __init__(self) -> None:
        self.objects = {
            "schedule/game_date=2026-10-23/snapshots/a.injury.pdf": b"report one",
            "schedule/game_date=2026-10-24/snapshots/b.injury.pdf": b"report one",
            "schedule/game_date=2026-10-24/snapshots/c.injury.pdf": b"report two",
            "schedule/injury-reports/sha256=ignored.pdf": b"not legacy",
        }
        self.uploads: list[str] = []

    def list_objects_v2(self, *, Bucket, Prefix, ContinuationToken=None):
        assert Bucket == "bucket"
        assert Prefix == "schedule/"
        assert ContinuationToken is None
        return {
            "IsTruncated": False,
            "Contents": [
                {
                    "Key": key,
                    "Size": len(payload),
                    "ETag": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
                }
                for key, payload in self.objects.items()
            ],
        }

    def get_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        return {"Body": BytesIO(self.objects[Key])}

    def head_object(self, *, Bucket, Key):
        assert Bucket == "bucket"
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        payload = self.objects[Key]
        digest = hashlib.sha256(payload).hexdigest()
        return {
            "ContentLength": len(payload),
            "Metadata": {"sha256": digest},
        }

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        assert Bucket == "bucket"
        assert kwargs["Metadata"]["sha256"] == hashlib.sha256(Body).hexdigest()
        self.objects[Key] = Body
        self.uploads.append(Key)


def test_backfill_deduplicates_legacy_pdfs_by_content() -> None:
    s3 = FakeS3()

    summary = backfill_legacy_injury_pdfs(
        bucket="bucket", prefix="schedule", execute=True, client=s3
    )
    repeated = backfill_legacy_injury_pdfs(
        bucket="bucket", prefix="schedule", execute=True, client=s3
    )

    assert summary.legacy_objects == 3
    assert summary.unique_legacy_payloads == 2
    assert summary.checksum_objects_created == 2
    assert len(s3.uploads) == 2
    assert repeated.checksum_objects_created == 0
    assert repeated.checksum_objects_existing == 2


def test_backfill_dry_run_does_not_upload() -> None:
    s3 = FakeS3()

    summary = backfill_legacy_injury_pdfs(
        bucket="bucket", prefix="schedule", client=s3
    )

    assert summary.executed is False
    assert summary.unique_legacy_payloads == 2
    assert summary.checksum_objects_created == 0
    assert s3.uploads == []
