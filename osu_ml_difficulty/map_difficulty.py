import tensorflow as tf
import scipy.optimize as opt
import numpy as np
import pandas as pd

from .dataset import augment_beatmap_data
from .raleigh import CDF
from .beatmap import MapData, lib



def evaluate_map(model, map_data):
    augmented = augment_beatmap_data(map_data)
    return np.maximum(model.predict(augmented)+10,0)


def fc_probability(difficulties, skill):
    return np.prod(CDF(x=1, sigma=difficulties*(1/skill)))


def get_map_required_skill(difficulties, probability_threshold=0.05): 
    return opt.root_scalar(
        lambda skill: fc_probability(difficulties, skill)-probability_threshold,
        bracket=(0.1,200),
        xtol=0.00001,
        rtol=0.00001
        )

mod_names = {
    "dt": "double_time",
    "hd": "hidden",
    "hr": "hard_rock",
    "ez": "easy",
    "ht": "half_time",
}

def map_required_skills_from_csv(map_csv, model_path="model.keras"):
    model = tf.keras.models.load_model(model_path)

    maplist = pd.read_csv(map_csv)
    maplist["Mods"].fillna("", inplace=True)
    print("id","name","mods","difficulty","peak_difficulty","ppv2_aim",sep=";")
    for _, row in maplist.iterrows():
        try:
            beatmap_id = row["ID"]
            mods = {m:True for m in row["Mods"].split(" ") if m}
            beatmap = lib.lookup_by_id(beatmap_id)
            map_data = MapData.from_beatmap(beatmap,**mods)

            note_difficulties = evaluate_map(model, map_data)
            skill = get_map_required_skill(note_difficulties)
            ppv2_aim = beatmap.aim_stars(**{mod_names[m]: True for m in mods if m != "hd"})
            print(beatmap_id,map_data.beatmap_name.replace(";",","),row["Mods"],skill.root,np.max(note_difficulties),ppv2_aim,sep=";")
        except KeyError:
            pass
