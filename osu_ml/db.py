import os

import numpy as np
import peewee as pw
from slider.mod import Mod

from  osu_ml import config




db = pw.SqliteDatabase(
    os.path.join(config.DATA_PATH,'replays.db'),
    pragmas={
        #'journal_mode': 'wal',
        'cache_size': -1 * 64000,  # 64MB
        'foreign_keys': 1,
        'ignore_check_constraints': 0,
        'synchronous': 0
    })

class BaseModel(pw.Model):
    class Meta:
        database = db

class User(BaseModel):
    username = pw.CharField(unique=True, max_length=20)
    rank = pw.IntegerField()
    pp = pw.FloatField()
    play_time = pw.FloatField()
    play_count = pw.FloatField()
    history_completeness = pw.SmallIntegerField()

    def batch_count(self):
        return self.replays.count() // config.REPLAYS_PER_BATCH

    def replay_batch(self, i, batch_size=config.REPLAYS_PER_BATCH):
        return replay_batch(self.id, i, batch_size)

def replay_batch(user_id, i, batch_size=config.REPLAYS_PER_BATCH):
    return Replay.select().where(Replay.user == user_id).order_by(Replay.timestamp).paginate(i, batch_size)


class Replay(BaseModel):
    user = pw.ForeignKeyField(User, backref="replays")

    mods = pw.BitField()
    ez = mods.flag(Mod.easy)
    hr = mods.flag(Mod.hard_rock)
    ht = mods.flag(Mod.half_time)
    dt = mods.flag(Mod.double_time)
    hd = mods.flag(Mod.hidden)
    fl = mods.flag(Mod.flashlight)
    so = mods.flag(Mod.spun_out)
    nf = mods.flag(Mod.no_fail)
    scorev2 = mods.flag(Mod.scoreV2)

    beatmap_md5 = pw.CharField(32)
    timestamp = pw.TimestampField(resolution=1e6, utc=True)
    filename = pw.CharField()
    count_300 = pw.IntegerField()
    count_100 = pw.IntegerField()
    count_50 = pw.IntegerField()
    count_miss = pw.IntegerField()
    max_combo = pw.IntegerField()
    score = pw.FloatField()

    def load_frames(self):
        return np.load(os.path.join(config.REPLAY_PATH,self.filename))

    def feature_path(self):
        return os.path.join(config.REPLAY_FEATURE_PATH, self.filename)

    def load_features(self):
        return np.load(self.feature_path())

Replay.add_index(Replay.user, Replay.timestamp)

def init_db():
    with db:
        db.create_tables([User, Replay])
