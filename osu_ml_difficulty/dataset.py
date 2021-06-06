from enum import IntEnum

import numpy as np
import tensorflow as tf
from .beatmap import beatmap_from_replay
from . import db, frame



class AugmentedBeatmapColumns(IntEnum):
    DELTA_T = 0
    FREQUENCY = 1
    DISTANCE = 2
    VELOCITY = 3
    DELTA_X = 4
    DELTA_Y = 5
    N_COLUMNS = 6

def difficulty_estimate(dist, time):
    return 0.1 * (dist / time)


def linear_fit(x, y):

    # discard 5% either side
    outlier_ub = int(len(x)*0.95)
    outlier_lb = int(len(x)*0.05)
    count = outlier_ub-outlier_lb

    outlier_filter = np.argpartition(y,[outlier_lb, outlier_ub])[outlier_lb:outlier_ub] 
    m, c = np.linalg.lstsq(np.vstack([y[outlier_filter],np.ones(count)]).T,x[outlier_filter], rcond=None)[0]

    return m, c

def augment_beatmap_data(map_data):
    pos = map_data.hit_objects[:,frame.POS] * map_data.scale
    time = map_data.hit_objects[:,frame.TIME] * 1e-3
    delta_pos = np.diff(pos, axis=0)
    delta_t = np.maximum(np.diff(time, axis=0),0.001)
    freq = 1 / delta_t
    distances = np.linalg.norm(delta_pos, axis=-1)
    zero_distance_mask = np.abs(distances) < 0.05
    directions = delta_pos / distances[:,None]
    directions[zero_distance_mask] = np.array([[1,0]])
    perpendicular_direction = directions @ np.array([[0,1],[-1,0]])

    # distance in same direction as previous movement
    relative_delta_x = np.einsum("...i,...i", directions[:-1], delta_pos[1:])

    # distance perpendicular to previous movement
    relative_delta_y = np.einsum("...i,...i", perpendicular_direction[:-1], delta_pos[1:])


    data = np.zeros((delta_t.shape[0], 5,AugmentedBeatmapColumns.N_COLUMNS), dtype="float32")
    data[:, 2, AugmentedBeatmapColumns.DELTA_T] = delta_t
    data[:, 2, AugmentedBeatmapColumns.FREQUENCY] = freq
    data[:, 2, AugmentedBeatmapColumns.DISTANCE] = distances
    data[:, 2, AugmentedBeatmapColumns.VELOCITY] = distances * freq

    data[1:, 2, AugmentedBeatmapColumns.DELTA_X] = relative_delta_x
    data[1:, 2, AugmentedBeatmapColumns.DELTA_Y] =  relative_delta_y

    # add data for previous 2 and next 2
    data[2:,0,:] = data[:-2,2,:]
    data[1:,1,:] = data[:-1,2,:]
    data[:-1,3,:] = data[1:,2,:]
    data[:-2,4,:] = data[2:,2,:]

    return data

def user_batch_dataset(user_id, batch):

    with db.db:
        replays = [r for r in db.replay_batch(user_id, batch)]
    
    hit_object_list = []
    hit_error_list = []
    for r in replays:
        try:
            replay_features = r.load_features()

            hit_object_data = augment_beatmap_data(beatmap_from_replay(r))
            hit_errors = np.linalg.norm(replay_features[1:,frame.POS], axis=-1) # skip first hit object - no delta_t or delta_pos

            hit_object_list.append(hit_object_data)
            hit_error_list.append(hit_errors)
        except FileNotFoundError:
            pass



    if hit_error_list:
        hit_object_data = np.concatenate(hit_object_list)
        hit_errors = np.concatenate(hit_error_list)
        # filter out large errors and nans
        # could be aiming somewhere else, misread, or missed clicking
        mask = hit_errors < 3

        hit_object_data = hit_object_data[mask]
        hit_errors = hit_errors[mask]

        hit_object_difficulty = difficulty_estimate(hit_object_data[:,2,AugmentedBeatmapColumns.DISTANCE],hit_object_data[:,2,AugmentedBeatmapColumns.DELTA_T])
        #hit_object_difficulty = hit_object_data[:,2,AugmentedBeatmapColumns.FITTS_DIFFICULTY]
        m, c = linear_fit(hit_errors, hit_object_difficulty)
        implied_difficulty = (hit_errors-c)/m
        return (hit_object_data, implied_difficulty)
    return (tf.constant([],shape=(0,5,AugmentedBeatmapColumns.N_COLUMNS), dtype="float32"), tf.constant([], shape=(0,),dtype="float32"))

def wrapped_user_batch_dataset(user_batch):
    def func(user_batch):
        return user_batch_dataset(int(user_batch[0]), int(user_batch[1]))

    py_func = tf.py_function(
        func,
        [user_batch],
        [tf.float32, tf.float32]
    )

    return (
        tf.ensure_shape(py_func[0], (None,5,AugmentedBeatmapColumns.N_COLUMNS)),
        tf.ensure_shape(py_func[1], (None,))
    )


def _make_dataset(user_batches):
    tf.random.shuffle(user_batches)

    dataset =  tf.data.Dataset.from_tensor_slices(user_batches)
            
    return dataset.interleave(
        lambda x: tf.data.Dataset.from_tensor_slices(wrapped_user_batch_dataset(x)),
        cycle_length=6,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE
        )


def make_training_dataset():
    """
    difficulty_func: a function to assess difficulty of hit objects - used to assess skill of user
    """

    with db.db:
        user_batches = [ [user.id, batch] for user in db.User.select() for batch in range(user.batch_count()) if batch % 10 != 0]

    return _make_dataset(user_batches)

def make_validation_dataset():
    """
    difficulty_func: a function to assess difficulty of hit objects - used to assess skill of user
    """
    with db.db:
        user_batches = [ [user.id, batch] for user in db.User.select() for batch in range(user.batch_count()) if batch % 10 == 0]
    
    return _make_dataset(user_batches)
