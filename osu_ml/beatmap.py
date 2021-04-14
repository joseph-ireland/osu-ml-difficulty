import os
from functools import lru_cache

import numpy as np
import slider
from slider.mod import circle_radius

from osu_ml import config
from osu_ml.pickle_memoize import pickle_memoize
from osu_ml import frame


lib = slider.Library(config.OSU_MAP_PATH, cache=128)


class MapData:
    __slots__=("hit_objects", "scale", "beatmap_id", "beatmap_name", "dt", "hd", "hr", "ez", "ht")

    def __init__(self, hit_objects, scale, beatmap_id, beatmap_name, dt, hd, hr, ez, ht):
        self.hit_objects = hit_objects
        self.scale = scale
        self.beatmap_id = beatmap_id
        self.beatmap_name = beatmap_name
        self.dt = dt
        self.hd = hd
        self.hr = hr
        self.ez = ez
        self.ht = ht

    def scaled_hit_objects(self):
        result = np.copy(self.hit_objects)
        result[:, frame.POS] *= self.scale
        result[:, frame.TIME] *= 1e-3
        return result

    @classmethod
    def from_beatmap(cls, beatmap: slider.Beatmap, dt, hd, hr, ez, ht):
        return cls(
            hit_objects = np.array([
                (h.time.total_seconds()*1e3, h.position.x, h.position.y)
                for h in beatmap.hit_objects(
                    spinners=False, hard_rock=hr, double_time=dt, half_time=ht
                )
            ], dtype="f"),
            scale=1/circle_radius(beatmap.cs(easy=ez, hard_rock=hr)),
            beatmap_id=beatmap.beatmap_id,
            beatmap_name=beatmap.display_name,
            dt=dt,
            hd=hd,
            hr=hr,
            ez=ez,
            ht=ht
        )


def beatmmap_pickle_path(beatmap_md5, **kwargs):
    mods = "".join(sorted([k for k,v in kwargs.items() if v]))
    return os.path.join(config.pkl_map_path, f"{beatmap_md5}[{mods}].pkl")

@lru_cache(256)
@pickle_memoize(beatmmap_pickle_path)
def get_beatmap(beatmap_md5, **kwargs):
    return MapData.from_beatmap(lib.lookup_by_md5(beatmap_md5),**kwargs)

def beatmap_from_replay(replay):
    return get_beatmap(
        replay.beatmap_md5,
        dt=replay.dt, hd=replay.hd, hr=replay.hr, ez=replay.ez, ht=replay.ht
    )
