from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.compat.v1 import Session, ConfigProto, GPUOptions

from tensorflow import keras
from tensorflow.keras import layers

from . import dataset

# don't crash trying to allocate all the gpu memory when other stuff can be using it
tf_config = ConfigProto(gpu_options=GPUOptions(allow_growth=True))
session = Session(config=tf_config)
set_session(session)


def create_model():
    inputs = keras.Input(shape=(4,dataset.AugmentedBeatmapColumns.N_COLUMNS,))
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


def fit():
    m = create_model()
    m.summary()
    training_data = dataset.make_training_dataset().shuffle(10000).batch(1024).prefetch(10)
    validation_data = dataset.make_validation_dataset().batch(1024).prefetch(10)
    
    history = m.fit(
        training_data,
        validation_data=validation_data,
        epochs=1,
        callbacks=[
            keras.callbacks.TensorBoard(update_freq=1000, histogram_freq=1000, profile_batch=(2,4))
        ])
    print(history.history)
    m.save("model.keras")

if __name__=="__main__":
    fit()