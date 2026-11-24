import json
from datetime import UTC, datetime
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlparse

import pytest
from botocore.exceptions import ClientError

from gestalt_pipelines.resources import NBAPlayerNamesResource, player_names
from gestalt_pipelines.rosters.resources import NBARosterArchive


@pytest.fixture
def saved(monkeypatch):
    objects = {}

    def read(*, Bucket, Key):
        assert Bucket == "nba-gestalt-data"
        if Key not in objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        raw = objects[Key]
        return {"Body": BytesIO(raw), "ETag": sha256(raw).hexdigest(),
                "LastModified": datetime(2026, 11, 12, tzinfo=UTC)}

    def paginator(_):
        return SimpleNamespace(paginate=lambda **kw: [{"Contents": [
            {"Key": key} for key in objects if key.startswith(kw["Prefix"])
        ]}])

    monkeypatch.setattr(player_names.boto3, "client", lambda _: SimpleNamespace(
        get_object=read, get_paginator=paginator,
    ))
    return objects


def index(saved, rows):
    raw = json.dumps({"resultSets": [{"name": "PlayerIndex", "headers": [
        "PERSON_ID", "PLAYER_FIRST_NAME", "PLAYER_LAST_NAME",
    ], "rowSet": rows}]}).encode()
    url = "https://stats.nba.com/stats/playerindex?Season=2026-27&Historical=1&LeagueID=00&TeamID=0"
    saved["lake/raw/playerindex/2026-27.json"] = raw
    saved["lake/raw/playerindex/2026-27.meta.json"] = json.dumps({
        "endpoint": "playerindex", "season": "2026-27", "url": url,
        "params": dict(parse_qsl(urlparse(url).query)), "sha256": sha256(raw).hexdigest(),
        "fetched_at": None, "acquisition_method": "manual_browser_response",
    }).encode()


def test_shared_resource_reads_uploaded_index_without_writing_or_fetching(saved):
    index(saved, [[1643416, "Chris", "Cenac Jr."], [1631250, "Pete", "Nance"]])
    before = dict(saved)
    resource = NBAPlayerNamesResource(season="2026-27")
    names, reference = resource.load()
    assert names == {1643416: "Chris Cenac Jr.", 1631250: "Pete Nance"}
    assert reference["sha256"] == sha256(saved["lake/raw/playerindex/2026-27.json"]).hexdigest()
    assert reference["metadata"]["uri"].endswith("2026-27.meta.json")
    assert saved == before
    index(saved, [[1643416, "Christopher", "Cenac Jr."]])
    assert resource.load()[0][1643416] == "Christopher Cenac Jr."


@pytest.mark.parametrize("bad", ["checksum", "source", "season", "empty", "conflict"])
def test_invalid_index_fails(saved, bad):
    rows = [[1, "NBA", "Player"]]
    if bad == "empty":
        rows = []
    if bad == "conflict":
        rows.append([1, "Different", "Name"])
    index(saved, rows)
    key = "lake/raw/playerindex/2026-27.meta.json"
    meta = json.loads(saved[key])
    if bad == "checksum":
        meta["sha256"] = "0"*64
    elif bad == "source":
        meta["url"] = meta["url"].replace("stats.nba.com", "example.com")
    elif bad == "season":
        meta["season"] = "2025-26"
    saved[key] = json.dumps(meta).encode()
    with pytest.raises(ValueError):
        NBAPlayerNamesResource(season="2026-27").load()


def put_roster(saved, day, teams):
    prefix = f"lake/artifacts/rosters/season=2026-27/game_date_pacific={day}/"
    record = {"schema_version": 1, "season": "2026-27", "game_date_pacific": day, "teams": teams}
    raw = json.dumps(record).encode()
    digest = sha256(raw).hexdigest()
    key = prefix+digest+".json"
    saved[key] = raw
    ref = {"uri": "s3://nba-gestalt-data/"+key, "sha256": digest}
    saved[prefix+"latest.json"] = json.dumps({**record, **ref}).encode()
    return ref


def test_roster_requires_own_date_latest_and_verifies_its_hash(saved):
    archive = NBARosterArchive()
    put_roster(saved, "2026-11-11", {"1": [101]})
    with pytest.raises(FileNotFoundError, match="2026-11-12/latest.json"):
        archive.load("2026-27", "2026-11-12")
    ref = put_roster(saved, "2026-11-12", {"1": [102]})
    record, actual = archive.load("2026-27", "2026-11-12")
    assert record["teams"] == {"1": [102]} and actual == ref
    key = ref["uri"].removeprefix("s3://nba-gestalt-data/")
    saved[key] += b" "
    with pytest.raises(ValueError, match="checksum"):
        archive.load("2026-27", "2026-11-12")


def test_latest_roster_uses_previous_date_until_target_exists_and_never_future(saved):
    from datetime import date
    archive = NBARosterArchive()
    put_roster(saved, "2026-11-14", {"1": [999]})
    target = date(2026, 11, 13)
    with pytest.raises(FileNotFoundError, match="No accepted roster"):
        archive.latest_selection("2026-27", target)
    prior = put_roster(saved, "2026-11-12", {"1": [101]})
    assert archive.latest_selection("2026-27", target) == ("2026-11-12", prior)
    same = put_roster(saved, "2026-11-13", {"1": [102]})
    assert archive.latest_selection("2026-27", target) == ("2026-11-13", same)
    assert archive.latest_selection("2026-27", date(2026, 11, 12)) == ("2026-11-12", prior)


def transactions(saved,day,rows,**changes):
    key=f'lake/raw/nba/transactions/season=2026-27/game_date_pacific={day}/latest.json'
    value={'season':'2026-27','game_date_pacific':day,'source_url':player_names.TRANSACTIONS_URL,
        'transactions':[{'PLAYER_ID':1643572.0,'PLAYER_SLUG':'rafael-castro','TRANSACTION_DATE':day+'T00:00:00',
          'TRANSACTION_DESCRIPTION':'Atlanta Hawks signed center Rafael Castro to a Two-Way Contract.',
          'first_observed_at':'2026-11-17T22:00:05.713371+00:00',**row} for row in rows],**changes}
    saved[key]=json.dumps(value).encode();return key


def test_transaction_signing_immediately_supplies_name_and_roster_match_without_repairs(saved):
    from datetime import date
    from gestalt_pipelines.injury_reports.utils import reconcile_games
    index(saved,[[1,'Existing','Player']]);resource=NBAPlayerNamesResource(season='2026-27')
    assert 1643572 not in resource.load()[0]
    key=transactions(saved,'2026-11-16',[{}]);before=dict(saved)
    names,ref=resource.load()
    assert names[1643572]=='Rafael Castro' and names[1]=='Existing Player'
    assert ref['transaction_player_ids']==[1643572] and ref['unresolved_transaction_player_ids']==[]
    assert ref['transactions']['2026-11-16']=={'uri':'s3://nba-gestalt-data/'+key,'sha256':sha256(saved[key]).hexdigest()}
    game={'game_id':'castro-game','game_date':date(2026,11,18),'game_time_utc':datetime(2026,11,19,1,tzinfo=UTC),
        'away_team_id':1610612737,'home_team_id':1610612760,'away_team_tricode':'ATL','home_team_tricode':'OKC'}
    report={'published_at':'2026-11-17T23:00:00+00:00','pdf_uri':'s3://fixture/report.pdf',
        'game_team_reports':{'2026-11-18|ATL@OKC':{}},'players':[{'game_date':'2026-11-18','matchup':'ATL@OKC',
        'team':'ATL','player_name':'Rafael Castro','status':'Doubtful','reason':'G League - Two-Way'}]}
    result=reconcile_games([game],{1610612737:{1643572:names[1643572]},1610612760:{}},report,{},names,set(ref['transaction_player_ids']))
    row=result['castro-game']['teams']['1610612737']['players']['1643572']
    assert row['identity_source']=='roster' and row['name_source']=='nba_transactions'
    assert result['castro-game']['complete'] and saved==before
    off_roster=reconcile_games([game],{1610612737:{},1610612760:{}},report,{},names,set(ref['transaction_player_ids']))
    known=off_roster['castro-game']['teams']['1610612737']['players']['1643572']
    assert known['identity_source']=='nba_transactions' and known['identity_warning']=='not_on_selected_roster'
    from gestalt_pipelines.injury_reports.utils import identity_rows, write_identity_warnings
    events={}
    def save_json(key,record):
        events[key]=record;return {'uri':'s3://nba-gestalt-data/'+key,'sha256':sha256(json.dumps(record).encode()).hexdigest()}
    archive=SimpleNamespace(season='2026-27',save_json=save_json)
    write_identity_warnings(archive,identity_rows(off_roster),occurred_at=datetime(2026,11,17,23,tzinfo=UTC),
        run_id='fixture',game_date='2026-11-18',extractions={'castro-game':{'uri':'s3://fixture/report','sha256':'a'*64}},artifact_path='s3://fixture/statuses',inputs={})
    assert 'official NBA transactions' in next(iter(events.values()))['message']


def test_playerindex_remains_authoritative_and_unused_transactions_are_not_dependencies(saved):
    index(saved,[[1643572,'Rafael','Castro']]);transactions(saved,'2026-11-16',[{'TRANSACTION_DESCRIPTION':'Different spelling'}])
    names,ref=NBAPlayerNamesResource(season='2026-27').load()
    assert names[1643572]=='Rafael Castro' and ref['transactions']=={} and ref['transaction_player_ids']==[]


def test_missing_official_name_is_explicit_without_inventing_slug_display_name(saved):
    index(saved,[[1,'Existing','Player']]);transactions(saved,'2026-11-16',[{'TRANSACTION_DESCRIPTION':'An incomplete official description.'}])
    names,ref=NBAPlayerNamesResource(season='2026-27').load()
    assert names=={1:'Existing Player'} and ref['unresolved_transaction_player_ids']==[1643572]


@pytest.mark.parametrize('fault',['source','season','date','observation','id'])
def test_invalid_transaction_identity_sources_are_rejected(saved,fault):
    index(saved,[[1,'Existing','Player']]);changes={};row={}
    if fault=='source':changes['source_url']='https://example.com/unapproved'
    if fault=='season':changes['season']='2025-26'
    if fault=='date':row['TRANSACTION_DATE']='2026-11-15T00:00:00'
    if fault=='observation':row['first_observed_at']='2026-11-17T22:00:00'
    if fault=='id':row['PLAYER_ID']=True
    transactions(saved,'2026-11-16',[row],**changes)
    with pytest.raises(ValueError):NBAPlayerNamesResource(season='2026-27').load()
