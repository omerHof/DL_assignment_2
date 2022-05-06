# region import the necessary packages
import numpy as np
import cv2
from random import shuffle
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.regularizers import l2, l1
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import tensorflow.keras.backend as K


# endregion

# region for colab
# from google.colab import drive
# drive.mount('/content/drive')
# import zipfile
# zip_ref = zipfile.ZipFile("/content/drive/MyDrive/lfw2.zip")
# zip_ref.extractall("lfw2_file")

# device_name = tensorflow.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#   raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))

# endregion

# region create dataset
def make_pairs2(examples_file):
    """
    This function read the data from train/test file.
    :param examples_file:
    :return:
    """
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pairImages = []
    pairLabels = []

    # open train text file
    ex = open(examples_file, "r")

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
    for line_arr in lines_arrs:
        if len(line_arr) == 3:  # the same images, label 1
            imgA = cv2.imread("File/lfw2/" + line_arr[0] + "/" + line_arr[0] + "_" + line_arr[1].zfill(4) + ".jpg",
                              0)
            imgB = cv2.imread("File/lfw2/" + line_arr[0] + "/" + line_arr[0] + "_" + line_arr[2].zfill(4) + ".jpg",
                              0)
            imgA = imgA / 255.0
            imgB = imgB / 255.0
            imgA = np.expand_dims(imgA, axis=-1)
            imgB = np.expand_dims(imgB, axis=-1)
            pairImages.append([imgA, imgB])
            pairLabels.append([1])
        elif len(line_arr) == 4:  # different images, label 2
            imgA = cv2.imread("File/lfw2/" + line_arr[0] + "/" + line_arr[0] + "_" + line_arr[1].zfill(4) + ".jpg",
                              0)
            imgB = cv2.imread("File/lfw2/" + line_arr[2] + "/" + line_arr[2] + "_" + line_arr[3].zfill(4) + ".jpg",
                              0)
            imgA = imgA / 255.0
            imgB = imgB / 255.0
            imgA = np.expand_dims(imgA, axis=-1)
            imgB = np.expand_dims(imgB, axis=-1)
            pairImages.append([imgA, imgB])
            pairLabels.append([0])
        else:
            pass
    ex.close()
    print('end')
    return np.array(pairImages), np.array(pairLabels)


print("[INFO] preparing positive and negative pairs...")
(pairTrain, labelTrain) = make_pairs2("File/Train.txt")
(pairTest, labelTest) = make_pairs2("File/Test.txt")

print(pairTrain.shape)
print(pairTest.shape)

# endregion

# region configurations
IMG_SHAPE = (250, 250, 1)
BATCH_SIZE = 32
EPOCHS = 2

# configure the siamese network
print("[INFO] building siamese network...")
W_init_1 = RandomNormal(mean=0, stddev=0.01, seed=420)
b_init = RandomNormal(mean=0.5, stddev=0.01, seed=420)
W_init_2 = RandomNormal(mean=0, stddev=0.02, seed=420)

left_input = Input(shape=IMG_SHAPE)
right_input = Input(shape=IMG_SHAPE)

convnet = Sequential([
    Conv2D(64, (10, 10), activation="relu", input_shape=IMG_SHAPE, kernel_initializer=W_init_1,
           bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(128, (7, 7), activation='relu', kernel_initializer=W_init_1,
           bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(128, (4, 4), activation='relu', kernel_initializer=W_init_1,
           bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init_1,
           bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(256, (4, 4), activation='relu', kernel_initializer=W_init_1,
           bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.3),
    Conv2D(512, (4, 4), activation='relu', kernel_initializer=W_init_1,
           bias_initializer=b_init, kernel_regularizer=l2(1e-2)),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(4096, activation="sigmoid", kernel_initializer=W_init_2,
          bias_initializer=b_init, kernel_regularizer=l2(1e-4))
])


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


# endregion

# region compile model
encoded_l = convnet(left_input)
encoded_r = convnet(right_input)

merge_layer = Lambda(euclidean_distance)([encoded_l, encoded_r])
# merge_layer  = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_l, encoded_r])
prediction = Dense(1, activation='sigmoid')(merge_layer)
model = Model(inputs=[left_input, right_input], outputs=prediction)

# optimizer= tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
optimizer = SGD(lr=0.01, momentum=0.5)
model.compile(loss="binary_crossentropy", optimizer=optimizer,
              metrics=["accuracy"])
print(model.summary())


# endregion

# region plot
def plot_training(H):
    """
    Define plot for history object that retuen from fit function
    :param H:
    :return:
    """
    # construct a plot that plots and saves the training history

    f, (ax1, ax2) = plt.subplots(2, 1)
    plt.style.use("ggplot")
    plt.figure()

    ax1.plot(H.history["accuracy"], label="train_acc")
    ax1.plot(H.history["val_accuracy"], label="val_acc")
    ax1.set(xlabel='Epoch', ylabel='Accuracy')
    ax1.legend(loc="lower left")

    ax2.plot(H.history["loss"], label="train_loss")
    ax2.plot(H.history["val_loss"], label="val_loss")
    ax2.set(xlabel='Epoch', ylabel='Loss')
    ax2.legend(loc="lower left")

    f.savefig("plot.png")


# endregion

# region main - train the model
print("[INFO] training model...")

with tensorflow.device('/device:GPU:0'):
    def scheduler(epoch, lr):
        if epoch < 50:
            return lr
        else:
            return lr * tensorflow.math.exp(-0.1)


    x_train, X_val, y_train, y_val = train_test_split(pairTrain, labelTrain, test_size=0.2, random_state=42)

    callback_lr = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)
    callback_early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    start_time = time.time()
    history = model.fit(
        [x_train[:, 0], x_train[:, 1]], y_train[:],
        validation_data=([X_val[:, 0], X_val[:, 1]], y_val[:]),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS, callbacks=[callback_lr, callback_early_stop], verbose=2)
    end_time = time.time()
    print(
        "epoch: {} epoch done: {} batch size: {} lr :{:.8f} momentum  final loss: {:.4f} final acc: {:.4f} final loss validation :{:.2f} %  "
        'final acc validation: {:.2f} % time_taken {:.2f} m'.format(EPOCHS, len(history.epoch), BATCH_SIZE,
                                                                    K.get_value(model.optimizer.learning_rate),
                                                                    # K.get_value(model.optimizer.momentum),
                                                                    history.history['loss'][len(history.epoch) - 1],
                                                                    history.history['accuracy'][len(history.epoch) - 1],
                                                                    history.history['val_loss'][len(history.epoch) - 1],
                                                                    history.history['val_accuracy'][
                                                                        len(history.epoch) - 1],
                                                                    (end_time - start_time) / 60))

prob = model.predict([pairTest[:, 0], pairTest[:, 1]])

fpr, tpr, thresholds = roc_curve(labelTest[:], prob)
accuracy_scores = []
for thresh in thresholds:
    accuracy_scores.append(accuracy_score(labelTest[:], [m > thresh for m in prob]))

accuracies = np.array(accuracy_scores)
max_accuracy = accuracies.max()
max_accuracy_threshold = thresholds[accuracies.argmax()]
print(f"Accuracy score test: {max_accuracy}")
print(f"Max Threshold: {max_accuracy_threshold}")

# plot the training history
print("[INFO] plotting training history...")
plot_training(history)

# endregion
