"""Synthetic accepted NBA inputs; no real S3 access."""
import io
from datetime import date, datetime, UTC
from hashlib import sha256
import pytest
from botocore.exceptions import ClientError
from gestalt_pipelines.resources import NBAScheduleResource
from gestalt_pipelines.team_rotation.resources import TeamRotationArchive
from gestalt_pipelines.team_rotation.utils import json_bytes
from gestalt_pipelines.utils import NBA_TEAM_IDS, PACIFIC

BUCKET="nba-gestalt-data"
FIRST=date(2026,10,20)


class Store:
    def __init__(self):
        self.objects={};self.writes=[];self.reads=[];self.hook=None
    def get_object(self,*,Bucket,Key):
        self.reads.append(Key)
        if Key not in self.objects:
            raise ClientError({"Error":{"Code":"NoSuchKey"}},"GetObject")
        raw=self.objects[Key]
        return {"Body":io.BytesIO(raw),"ETag":sha256(raw).hexdigest()}
    def put_object(self,*,Bucket,Key,Body,IfMatch=None,IfNoneMatch=None,**kw):
        assert Key.startswith(("lake/artifacts/team_rotation/","lake/artifacts/errors/pipeline=team_rotation/"))
        if self.hook:self.hook(Key)
        if (IfNoneMatch=="*" and Key in self.objects) or (IfMatch is not None and (Key not in self.objects or sha256(self.objects[Key]).hexdigest()!=IfMatch)):
            raise ClientError({"Error":{"Code":"PreconditionFailed"}},"PutObject")
        self.objects[Key]=Body;self.writes.append(Key)
        return {}


@pytest.fixture
def saved(monkeypatch):
    store=Store()
    monkeypatch.setattr("boto3.client",lambda *a,**kw:store)
    a=TeamRotationArchive()
    season="2026-27";day="2026-10-20"
    def save(prefix,doc):
        raw=json_bytes(doc);digest=sha256(raw).hexdigest();key=prefix+digest+".json"
        store.objects[key]=raw
        return {"uri":f"s3://{BUCKET}/{key}","sha256":digest}
    teams=[{"teamId":tid,"teamTricode":f"T{i:02}"} for i,tid in enumerate(NBA_TEAM_IDS)]
    schedule_doc={"leagueSchedule":{"seasonYear":season,"leagueId":"00","gameDates":[{"games":[
        {"gameId":f"0022600{i:03}","homeTeam":teams[i],"awayTeam":teams[(i+1)%30]} for i in range(30)
    ]}]}}
    schedule_ref={"uri":f"s3://{BUCKET}/lake/raw/scheduleleaguev2/season={season}/sha256=mock/schedule.json","sha256":"a"*64}
    schedule_key=f"lake/raw/scheduleleaguev2/season={season}/latest.json"
    store.objects[schedule_key]=json_bytes(schedule_ref)
    games=[];rows=[];refs={}
    for index in range(2):
        gid=f"002260000{index+1}"
        home,away=teams[index*2],teams[index*2+1]
        g={"game_id":gid,"game_time_utc":datetime(2026,10,21,1,tzinfo=UTC),"game_date":FIRST,
           "home_team_id":home["teamId"],"away_team_id":away["teamId"],
           "home_team_tricode":home["teamTricode"],"away_team_tricode":away["teamTricode"]}
        games.append(g)
        r={"game_id":gid,"stint_index":0,"home_team_id":home["teamId"],"away_team_id":away["teamId"],
           "home_player_ids":list(range(index*10+1,index*10+6)),"away_player_ids":list(range(index*10+6,index*10+11)),
           "possessions":10.123456789,"duration_seconds":120}
        rows.append(r)
        box={"game":{"gameId":gid,"gameStatus":3}}
        for side,t in (("home",home),("away",away)):
            box["game"][side+"Team"]={"teamId":t["teamId"],"players":[
                {"personId":pid,"name":f"Player {pid}"} for pid in r[side+"_player_ids"]]}
        ref=save(f"lake/raw/cdn/boxscore/season={season}/game_id={gid}/",box)
        key=ref["uri"].removeprefix(f"s3://{BUCKET}/").removesuffix(".json")+".meta.json"
        raw=json_bytes({"source":"official NBA"});store.objects[key]=raw
        ref.update(metadata_uri=f"s3://{BUCKET}/{key}",metadata_sha256=sha256(raw).hexdigest())
        refs[gid]={"box_score":ref}
    doc={"schema_version":1,"season":season,"game_date_pacific":day,"processing_version":"v1",
         "validation":{"complete":True,"expected_game_ids":[g["game_id"] for g in games],"processed_game_ids":[g["game_id"] for g in games]},
         "inputs":{"schedule":schedule_ref,"games":refs},"rows":rows}
    prefix=f"lake/artifacts/stints/season={season}/game_date_pacific={day}/"
    def accept():
        pointer={**save(prefix,doc),"season":season,"game_date_pacific":day,"processing_version":"v1","schema_version":1}
        store.objects[prefix+"latest.json"]=json_bytes(pointer)
        return pointer
    accept()
    monkeypatch.setattr(NBAScheduleResource,"load",lambda self,reference=None:(json_bytes(schedule_doc),schedule_ref))
    monkeypatch.setattr(NBAScheduleResource,"games_between",lambda self,first,last,reference=None,allow_empty=False:([g for g in games if first<=g["game_time_utc"].astimezone(PACIFIC).date()<=last],schedule_ref))
    return a,store,NBAScheduleResource(season=season),doc,accept,save,games
