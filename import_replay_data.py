"""
Import replay files and store in binary format
"""

import os
from concurrent.futures import ProcessPoolExecutor
import logging

from peewee import fn
import numpy as np
import pandas as pd

from osu_ml_difficulty import db, config
from osu_ml_difficulty.fast_replay import FastReplay, GameModeNotSupported

USERS_FILENAME = os.path.join(config.DOWNLOAD_PATH, "users.csv")


def import_users():
    responses = pd.read_csv(USERS_FILENAME)

    db.init_db()

    with db.db:
        for user in db.User.select():
            if user.username != responses["username"][user.id]:
                raise RuntimeError("Mismatch between database and response form")

        next_row = db.User.select().count()

        for i, row in responses[next_row:].iterrows():
            user = db.User.create(
                id=i,
                username=row["username"],
                rank=row["rank"],
                pp=row["pp"],
                play_time=row["play_time"],
                play_count=row["play_count"],
                history_completeness=row["history_completeness"]
                )

        pending_users = [u for u in db.User.select().where(
            ~fn.EXISTS(
                db.Replay.select(1).where(db.User.id == db.Replay.user)
            )
        )]

    return pending_users, responses

def import_replays():
    pending_users, _ = import_users()

    for user in pending_users:
        parse_replays(user)


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

def parse_replays(user, processes=4):
    path = os.path.join(config.DOWNLOAD_PATH, user.username)

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
    import_replays()