import time
from random import shuffle
import cv2
import numpy as np
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.models import Model, Sequential
from keras.layers import Flatten
from keras import initializers
# from keras.initializers import RandomNormal
from keras.layers import Lambda
from keras.regularizers import l2, l1
# from tensorflow.keras.optimizers import SGD
import keras.optimizers
import keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

params = {
    'IMG_SHAPE': (250, 250, 1),
    'BATCH_SIZE': 32,
    'EPOCHS': 2,
    'STRIDE': 1
}



def load_data(samples_txt_file):
    """
    Function that recieve text file of train or test, and arrange all the
    images in array.
    :param samples_txt_file:
    :return:
    """

    # open train text file
    ex = open(samples_txt_file, "r")

    # iterate over the lines and build the dataset
    lines_arrs = []
    lines = ex.readlines()
    for line in lines:
        line_arr = line.split()
        if len(line_arr) == 3:  # the same images, label 1
            lines_arrs.append(line_arr)
        elif len(line_arr) == 4:  # different images, label 2
            lines_arrs.append(line_arr)
        else:
            pass
    shuffle(lines_arrs)
    ex.close()
    return lines_arrs

def upload_pair(line_arr, same_person=True):
    imgA = cv2.imread(f'data/lfw2/{line_arr[0]}/{line_arr[0]}_'
                      f'{line_arr[1].zfill(4)}.jpg', 0)
    if same_person:
        imgB = cv2.imread(f'data/lfw2/{line_arr[0]}/{line_arr[0]}_'
                          f'{line_arr[2].zfill(4)}.jpg', 0)
    else:
        imgB = cv2.imread(f'data/lfw2/{line_arr[2]}/{line_arr[2]}_'
                          f'{line_arr[3].zfill(4)}.jpg', 0)
    imgA = imgA / 255.0
    imgB = imgB / 255.0
    imgA = np.expand_dims(imgA, axis=-1)
    imgB = np.expand_dims(imgB, axis=-1)
    return imgA,imgB


def make_pairs2(data_array):
    pair_images = []
    pair_labels = []
    for line_arr in data_array:
        if len(line_arr) == 3:  # the same images, label 1
            imgA, imgB = upload_pair(line_arr)
            pair_images.append([imgA, imgB])
            pair_labels.append(1)
        elif len(line_arr) == 4:  # different images, label 2
            imgA, imgB = upload_pair(line_arr,False)
            pair_images.append([imgA, imgB])
            pair_labels.append(0)
        else:
            pass
    return np.array(pair_images), np.array(pair_labels)

def explore_data():
    pass

def create_architecture():
    W_init_1 = initializers.RandomNormal(mean=0, stddev=0.01, seed=420)
    b_init = initializers.RandomNormal(mean=0.5, stddev=0.01, seed=420)
    W_init_2 = initializers.RandomNormal(mean=0, stddev=0.02, seed=420)
    convnet = Sequential([
        Conv2D(64, (10, 10), activation="relu", input_shape=params['IMG_SHAPE'],
               strides = params['STRIDE'], kernel_initializer=W_init_1,
               bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
        MaxPool2D(),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(128, (7, 7), activation='relu', kernel_initializer=W_init_1,
               strides = params['STRIDE'],
               bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
        MaxPool2D(),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init_1,
               strides =params['STRIDE'],
               bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
        MaxPool2D(),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init_1,
               strides =params['STRIDE'],
               bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
        MaxPool2D(),
        BatchNormalization(),
        Dropout(0.3),
        Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init_1,
               strides =params['STRIDE'],
               bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
        MaxPool2D(),
        BatchNormalization(),
        Dropout(0.3),
        # Conv2D(512, (4, 4), activation='relu', kernel_initializer=W_init_1,
        #        strides = params['STRIDE'],
        #        bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
        # BatchNormalization(),
        # Dropout(0.3),
        Flatten(),
        Dense(4096, activation="sigmoid", kernel_initializer=W_init_2,
              bias_initializer=b_init, kernel_regularizer=l2(1e-4))
    ])
    print(convnet.summary())
    return convnet

def euclidean_distance(vectors):
    """
    Function for define the distance between two image
    :param vectors:
    :return:
    """
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors

    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)

    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def absolute_distance(vectors):
    (vec1,vec2) = vectors
    return K.abs(vec1 - vec2)

def configure_model(convnet,left_input,right_input):
    # region compile model
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)
    merge_layer = Lambda(absolute_distance)([encoded_l, encoded_r])
    # merge_layer  = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_l, encoded_r])
    prediction = Dense(1, activation='sigmoid')(merge_layer)
    model = Model(inputs=[left_input, right_input], outputs=prediction)

    # optimizer= tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.5)
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])

    print(model.summary())
    return model


def get_callbacks():
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    filepath = "/models_checkpoints/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    rlop = ReduceLROnPlateau(patience=5)
    return [early_stopping, checkpoint, rlop]


def plot(train_results, val_results, title, y_label, batch_size):
    plt.plot([i[0] for i in train_results], [i[1] for i in train_results], label=f'training {y_label}')
    plt.plot([i[0] for i in val_results], [i[1] for i in val_results], label=f'validation {y_label}')
    plt.title(f'{title}: batch size:{batch_size}')
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def train_model(train_set):

    x_train, X_val, y_train, y_val = train_test_split(train_set[0], train_set[1], test_size=0.2, random_state=42)
    callbacks = get_callbacks()
    start_time = time.time()
    history = model.fit(
        [x_train[:, 0], x_train[:, 1]], y_train[:],
        validation_data=([X_val[:, 0], X_val[:, 1]], y_val[:]),
        batch_size=params["BATCH_SIZE"],
        epochs=params["EPOCHS"], callbacks=callbacks, verbose=2)
    end_time = time.time()
    print(
        "epoch: {} epoch done: {} batch size: {} lr :{:.8f} momentum  final loss: {:.4f} final acc: {:.4f} final loss validation :{:.2f} %  "
        'final acc validation: {:.2f} % time_taken {:.2f} m'.format(params["EPOCHS"], len(history.epoch), params["BATCH_SIZE"],
                                                                    K.get_value(model.optimizer.learning_rate),
                                                                    # K.get_value(model.optimizer.momentum),
                                                                    history.history['loss'][len(history.epoch) - 1],
                                                                    history.history['accuracy'][len(history.epoch) - 1],
                                                                    history.history['val_loss'][len(history.epoch) - 1],
                                                                    history.history['val_accuracy'][
                                                                        len(history.epoch) - 1],
                                                                    (end_time - start_time) / 60))
    plot(history.history['loss'], history.history['val_loss'], "Train validation loss", "Loss", params["BATCH_SIZE"])
    plot(history.history['accuracy'], history.history['val_accuracy'], "Train validation accuracy", "Accuracy", params["BATCH_SIZE"])

    return model


def evaluate_model(test_set, trained_model):
    return trained_model.predict([test_set[:, 0], test_set[:, 1]])


def show_results(results):
    pass


if __name__ == '__main__':
    train_txt_file = "data/pairsDevTrain.txt"
    test_txt_file = "data/pairsDevTest.txt"
    train_data_array = load_data(train_txt_file)
    test_data_array = load_data(test_txt_file)

    train_set = make_pairs2(train_data_array)
    test_set = make_pairs2(test_data_array)
    convnet = create_architecture()
    left_input = Input(shape=params['IMG_SHAPE'])
    right_input = Input(shape=params['IMG_SHAPE'])
    model = configure_model(convnet, left_input, right_input)

    trained_model = train_model(train_set)
    results = evaluate_model(test_set, trained_model)
    show_results(results)


