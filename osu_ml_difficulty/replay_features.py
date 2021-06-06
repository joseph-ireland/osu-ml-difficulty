import os
import logging

import numpy as np
import numba

from osu_ml_difficulty import db, beatmap
from osu_ml_difficulty import frame


@numba.njit(cache=True)
def hit_object_clicks(clicks, hit_objects):
    n_hit_objects = hit_objects.shape[0]
    result = np.full((hit_objects.shape[0], clicks.shape[1]), np.nan, dtype="f")
    TIME = frame.TIME

    j=0
    for click in clicks:
        t = click[TIME]
        # find nearest hit object (iterate until next object is further than current)
        while j + 1 < n_hit_objects and (
                    hit_objects[j+1,TIME] < t or
                    abs(hit_objects[j+1, TIME]-t) < abs(hit_objects[j, TIME]-t)):
            j += 1

        hit_object_time = hit_objects[j, TIME]
        nearest_time = abs(result[j, TIME] - t)
        if np.isnan(nearest_time) or nearest_time > abs(hit_object_time-t):
            result[j] = click

    return result



def replay_features(replay: db.Replay, map_data: beatmap.MapData):
    """
    Returns hit error and velocity for nearest click to each hit object
    """
    if map_data is None:
        return None

    clicks = replay.load_frames()

    if clicks.size == 0:
        return None

    if replay.dt:
        clicks[:,frame.TIME] *= 2/3
    elif replay.ht:
        clicks[:,frame.TIME] *= 4/3

    scale = map_data.scale
    hit_objects = map_data.hit_objects
    nearest_hit_object_click = hit_object_clicks(clicks, hit_objects)

    error = nearest_hit_object_click[:,0:3] - hit_objects
    error[:,frame.POS] *= scale

    return np.column_stack((error, nearest_hit_object_click[:,frame.V]))


def calculate_replay_features(replay: db.Replay):
    bmap = beatmap.beatmap_from_replay(replay)
    features = replay_features(replay, bmap)
    if features is not None:
        try:
            np.save(replay.feature_path(), features)
        except:
            os.remove(replay.feature_path())
            raise
    return features


def calculate_all_replay_features(force=False, skip_exceptions=True):
    with db.db:
        processed = 0
        progress = 0
        skipped = 0
        errored = 0
        total = db.Replay.select().count()
        print()
        for replay in db.Replay.select():
            progress += 1
            if force or not os.path.exists(replay.feature_path()):
                try:
                    if calculate_replay_features(replay) is not None:
                        processed += 1
                        if processed % 100 == 0:
                            print(
                                f"\r"
                                f"processed {processed} replays. "
                                f"Progress: {progress}/{total}. "
                                f"Couldn't find {skipped} beatmaps, and failed to process {errored}",
                                end=""
                            )

                except KeyError: # beatmap not found
                    skipped += 1
                except Exception:
                    if not skip_exceptions:
                        raise
                    errored += 1
                    logging.exception(
                        "Failed to parse replay: %s map: %s", 
                        replay,
                        replay.beatmap_md5
                    )

                    print(
                        f"\r"
                        f"processed {processed} replays. "
                        f"Progress: {progress}/{total}. "
                        f"Couldn't find {skipped} beatmaps, and failed to process {errored}",
                        end=""
                    )

if __name__ == "__main__":
    calculate_all_replay_features()
