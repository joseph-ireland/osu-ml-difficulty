from urllib.parse import urlparse, parse_qs
import io
import os
from concurrent.futures import ProcessPoolExecutor
import logging

import gspread
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from patoolib import extract_archive
from peewee import fn
import numpy as np

from osu_ml import db, config
from osu_ml.fast_replay import FastReplay, GameModeNotSupported


SHEET_URL = "https://docs.google.com/spreadsheets/d/1lpTCKNPoE2ikUN6liXjOcUgvnE7ePR8HDFD77YZe-rg/edit?usp=sharing"
CRED_PATH = "../secrets/replay_data_gdrive_credentials.json"
scope = ["https://spreadsheets.google.com/feeds", "https://googleapis.com/auth/drive"]

download_path = os.path.join(config.DATA_PATH, "replay_downloads")


creds = ServiceAccountCredentials.from_json_keyfile_name(CRED_PATH)
gdrive_service = build("drive", "v3", credentials=creds)
gspread_service = gspread.authorize(creds)

def download_replays(user, url):
    extract_path = os.path.join(download_path, user.username)
    if not os.path.exists(extract_path):
        fileId = parse_qs(urlparse(url).query)["id"][0]
        metadata = gdrive_service.files().get(fileId=fileId).execute()
        _, extension = os.path.splitext(metadata["name"])
        filepath = os.path.join(download_path, user.username + extension)

        if not os.path.isfile(filepath):
            request = gdrive_service.files().get_media(fileId=fileId)

            with io.FileIO(filepath+".tmp", mode='wb') as fh:
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                print(f"Downloading {filepath}")

                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Downloading {filepath}: {int(status.progress()*100)}%.")
            os.rename(filepath+".tmp", filepath)

        extract_archive(filepath, outdir=extract_path)

    parse_replays(extract_path, user)


def download_responses():
    responses = gspread_service.open_by_url(SHEET_URL).get_worksheet(0).get_all_records()

    db.init_db()

    with db.db:
        for user in db.User.select():
            if user.username != responses[user.id]["username"]:
                raise RuntimeError("Mismatch between database and response form")

        next_row = db.User.select().count()

        for i, row in enumerate(responses[next_row:], next_row):
            user = db.User.create(
                id=i,
                username=row["username"],
                rank=row["rank"],
                pp=row["pp"],
                play_time=row["play_time"],
                play_count=row["play_count"],
                history_completeness=row["history_completeness"]
                )

        users = [u for u in db.User.select().where(
            ~fn.EXISTS(
                db.Replay.select(1).where(db.User.id == db.Replay.user)
            )
        )]

    for user in users:
        replay_url = responses[user.id]["replays_url"]
        download_replays(user, replay_url)


def replay_paths(replay_dir):
    for directory, _, filenames in os.walk(replay_dir):
        for filename in filenames:
            if(filename.endswith(".osr")):
                yield os.path.join(directory, filename)

def parse_replay(replay_path):
    try:
        return FastReplay.from_path(replay_path)
    except GameModeNotSupported:
        return None
    except Exception:
        logging.exception("error parsing replay")
        return None

def parse_replays(path, user, processes=4):
    with db.db:
        i=0
        with ProcessPoolExecutor(processes) as executor:
            for replay in executor.map(parse_replay, replay_paths(path), chunksize=100):
                if replay is not None:
                    filename = f"{user.username}-{replay.beatmap_md5}-{replay.timestamp.strftime('%Y-%m-%d_%H-%M-%S.%f')}.npy"
                    np.save(os.path.join(config.REPLAY_PATH, filename), replay.actions)
                    db.Replay.create(
                        user=user,
                        mods=replay.difficulty_mods(),
                        beatmap_md5=replay.beatmap_md5,
                        timestamp=replay.timestamp,
                        filename=filename,
                        count_300=replay.count_300,
                        count_100=replay.count_100,
                        count_50=replay.count_50,
                        count_miss=replay.count_miss,
                        max_combo=replay.max_combo,
                        score=replay.score
                    )
                    i+=1
                    if i % 500 == 0:
                        print(f"Parsing replays for {user.username}: {i}")
            print(f"Parsing replays for {user.username}: {i}")


if __name__ == "__main__":
    download_responses()
