import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers

from . import dataset

# don't crash trying to allocate all gpu memory at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def create_model():
    inputs = keras.Input(shape=(5,dataset.AugmentedBeatmapColumns.N_COLUMNS,))
    x = layers.Flatten()(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1, activation="linear")(x)
    model = keras.Model(inputs=inputs,outputs=output, name="osu_difficulty_model")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0005), loss="mae")
    return model


def fit(filename="model.keras", max_batch_count=None):
    m = create_model()
    m.summary()
    training_data = dataset.make_training_dataset().shuffle(10000).batch(1024).prefetch(10)
    validation_data = dataset.make_validation_dataset().batch(1024).prefetch(10)
    
    if max_batch_count:
        training_data = training_data.take(max_batch_count)
        validation_data = validation_data.take(max_batch_count)

    history = m.fit(
        training_data,
        validation_data=validation_data,
        epochs=1,
        callbacks=[
            keras.callbacks.TensorBoard(update_freq=1000, histogram_freq=1000, profile_batch=(2,4))
        ])
    print(history.history)
    m.save(filename)


def load_model(path):
    return keras.models.load_model(path)

if __name__=="__main__":
    fit()