import os

DATA_PATH = "./data"
DOWNLOAD_PATH = os.path.join(DATA_PATH, "replay_downloads")
REPLAY_PATH = os.path.join(DATA_PATH, "replays")
REPLAY_FEATURE_PATH = os.path.join(DATA_PATH, "replay_features")
OSU_MAP_PATH = "../osu_files/"
pkl_map_path = os.path.join(DATA_PATH, "map_cache")

REPLAYS_PER_BATCH = 200
