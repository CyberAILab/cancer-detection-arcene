import numpy as np
from keras import regularizers
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import optimizers
from keras import callbacks
import arcene_dataset_load

np.random.seed(7)

#precision 85%
def create_model(verbose = False):
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=5, strides=2, padding='same', activation='relu', input_shape=(10000,1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=6, strides=2, padding='valid', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    model.add(Dense(1, activation='sigmoid'))

    sgd = optimizers.SGD(lr=0.001, clipnorm=1.)

    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    if(verbose):
        print(model.summary())
    return model

_, train_data_array, train_label_array, validation_data_array, validation_label_array = arcene_dataset_load.read_dataset()
train_data_array = np.expand_dims(train_data_array, axis=2)
validation_data_array = np.expand_dims(validation_data_array, axis=2)

model = create_model(True)

tbCallBack = callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=False)
model.fit(train_data_array, train_label_array, epochs = 100, batch_size = 1, callbacks=[tbCallBack])

scores = model.evaluate(validation_data_array, validation_label_array, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
model.save_weights("model_" + str(scores[1]) + "prec.h5")