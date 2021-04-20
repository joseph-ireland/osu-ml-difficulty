from enum import IntFlag, IntEnum
import io

import lzma
import numpy as np
import pandas as pd

from slider.replay import Replay
from slider.game_mode import GameMode
from slider.mod import Mod
from slider.utils import consume_byte, consume_short, consume_int, consume_string, consume_datetime

from . import frame

class GameModeNotSupported(RuntimeError):
    pass

class Buttons(IntFlag):
    """Bitmask values for keypresses.
    """
    M1 = 1
    M2 = 2
    K1 = 5
    K2 = 10


def _consume_actions(buffer):
    compressed_byte_count = consume_int(buffer)
    compressed_data = buffer[:compressed_byte_count]
    del buffer[:compressed_byte_count]
    decompressed_data = lzma.decompress(compressed_data)
    return _parse_actions(decompressed_data.decode("ascii"))

def _parse_actions(data):
    raw_actions = pd.read_csv(
        io.StringIO(data),
        delimiter="|",
        lineterminator=",",
        dtype="f",
        header=None
    ).to_numpy()

    if raw_actions.size != 0 and raw_actions[-1,0] == -12345:
        # actions can contain an element with time offset -12345 at the end
        # which is used to contain an RNG seed, we should ignore it
        np.delete(raw_actions, -1, axis=0)

    time_deltas = raw_actions[:, frame.TIME]
    positions = raw_actions[:, frame.POS]
    delta_pos = np.diff(positions, axis=0)
    delta_t = np.clip(time_deltas[:-1],1e-6, None)
    smooth_filter = np.array([0.25,0.5,0.25], dtype="f")
    v_x = np.convolve(delta_pos[:,0] / delta_t, smooth_filter, mode="same")
    v_y = np.convolve(delta_pos[:,1] / delta_t, smooth_filter, mode="same")

    times = np.cumsum(time_deltas)
    button_mask = raw_actions[:,3].astype(int)
    b1_pressed = (button_mask & Buttons.K1).astype(bool)
    b2_pressed = (button_mask & Buttons.K2).astype(bool)

    # true if either button1 or button2 is newly pressed in a frame
    click_mask = np.logical_or(b1_pressed[1:] > b1_pressed[:-1], b2_pressed[1:] > b2_pressed[:-1])

    #only keep rows where a button is clicked
    return np.column_stack((times[1:], positions[1:], v_x, v_y))[click_mask]



class FastReplay(Replay):
    """Lightweight replay class that only collects click positions, using pandas/numpy to speed up parsing
    """

    def difficulty_mods(self):
        return Mod.pack(**{
            mod : self.__getattribute__(mod)
            for mod in ("easy", "hard_rock", "half_time", "double_time",
                        "hidden", "flashlight", "spun_out", "no_fail", "scoreV2")
        })

    @classmethod
    def parse(cls, data, *_args, **_kwargs):
        """Parse a replay from ``.osr`` file data.

        Parameters
        ----------
        data : bytes
            The data from an ``.osr`` file.

        Returns
        -------
        replay : Replay
            The parsed replay.

        Raises
        ------
        ValueError
            Raised when ``data`` is not in the ``.osr`` format.
        """

        buffer = bytearray(data)

        mode = GameMode(consume_byte(buffer))
        if mode != GameMode.standard:
            raise GameModeNotSupported(mode)

        version = consume_int(buffer)
        beatmap_md5 = consume_string(buffer)
        player_name = consume_string(buffer)
        replay_md5 = consume_string(buffer)
        count_300 = consume_short(buffer)
        count_100 = consume_short(buffer)
        count_50 = consume_short(buffer)
        count_geki = consume_short(buffer)
        count_katu = consume_short(buffer)
        count_miss = consume_short(buffer)
        score = consume_int(buffer)
        max_combo = consume_short(buffer)
        full_combo = bool(consume_byte(buffer))
        mod_mask = consume_int(buffer)
        life_bar_graph = consume_string(buffer)
        timestamp = consume_datetime(buffer)
        actions = _consume_actions(buffer)

        mod_kwargs = Mod.unpack(mod_mask)
        # delete the alias field names
        del mod_kwargs['relax2']
        del mod_kwargs['last_mod']

        return cls(
            mode=mode,
            version=version,
            beatmap_md5=beatmap_md5,
            player_name=player_name,
            replay_md5=replay_md5,
            count_300=count_300,
            count_100=count_100,
            count_50=count_50,
            count_geki=count_geki,
            count_katu=count_katu,
            count_miss=count_miss,
            score=score,
            max_combo=max_combo,
            full_combo=full_combo,
            life_bar_graph=life_bar_graph,
            timestamp=timestamp,
            actions=actions,
            beatmap=None,
            **mod_kwargs,
        )
