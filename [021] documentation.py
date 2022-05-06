import os
import sys
import cv2
import math
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

""" 
Data Section
"""


def parse_pairs(same_len, train_or_test, train_file, test_file, images):
    """
    The method parse the train/test files and returns the pairs that representing the same and different human.
    :param same_len: Number of lines in the file of pairs that representing the same human.
    :param train_or_test: Flag indicates whether its train or test file.
    :param train_file: Name of train file.
    :param test_file: Name of test file.
    :param images: Name of images directory.
    :return: Lists of same and different pairs.
    """
    same = []
    different = []
    if train_or_test == "train":
        file = train_file
    else:
        file = test_file
    with open(file) as pairs_file:
        lines = [line.strip("\n").split("\t") for line in pairs_file]
        lines = lines[1:]
    for line in lines[:same_len]:
        image_file_1 = os.path.join(images, line[0], line[0] + "_{:04d}.jpg".format(int(line[1]))).replace("\\", "/")
        image_file_2 = os.path.join(images, line[0], line[0] + "_{:04d}.jpg".format(int(line[2]))).replace("\\", "/")
        same.append((image_file_1, image_file_2))
    for line in lines[same_len:]:
        image_file_1 = os.path.join(images, line[0], line[0] + "_{:04d}.jpg".format(int(line[1]))).replace("\\", "/")
        image_file_2 = os.path.join(images, line[2], line[2] + "_{:04d}.jpg".format(int(line[3]))).replace("\\", "/")
        different.append((image_file_1, image_file_2))
    return same, different


def read_images(combined_pairs_lists, crop=False):
    """
    The method receives the same and different lists, flatten them and read the images corresponding to the pairs.
    :param combined_pairs_lists: The same and different lists of image pairs.
    :param crop: boolean. Crop the image or not.
    :return: 2 numpy arrays, x1 contains the first image of each of the pairs, and x2 contains the second image of each
    of the pairs.
    """
    x1 = []
    x2 = []
    pairs_list = [pair for pair_list in combined_pairs_lists for pair in pair_list]
    for pair in pairs_list:
        image_1 = cv2.imread(pair[0], cv2.IMREAD_GRAYSCALE)
        image_1 = cv2.normalize(image_1, image_1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_1 = np.reshape(image_1, (250, 250, 1))
        if crop:
            image_1 = crop_image(image_1)
        x1.append(image_1)

        image_2 = cv2.imread(pair[1], cv2.IMREAD_GRAYSCALE)
        image_2 = cv2.normalize(image_2, image_2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image_2 = np.reshape(image_2, (250, 250, 1))
        if crop:
            image_2 = crop_image(image_2)
        x2.append(image_2)
    x1 = np.array(x1)
    x2 = np.array(x2)
    return x1, x2


def crop_image(image):
    """
    Crops the image to a new size around the image center.
    :param image: image file to crop.
    :return: the cropped image.
    """
    current_dim = image.shape[0]
    target_dim = 170
    left = int((current_dim - target_dim) / 2)
    right = current_dim - left
    cropped = image[left:right, left:right]
    return cropped


def data_loader(train_file, test_file, images, crop=False):
    """
    The method load the images, that are stored in "images" director, to train and test arrays according to the
    train_file and test_file. The method also create the labeles according to the directories of each paires
    (1 if they are from thesame directory and 0 if not).
    :param train_file: The train configuration file path
    :param test_file: The test configuration file path
    :param images: The main image directory
    :param crop: boolean. Crop the image of not.
    :return: data and labels for train and test.
    """
    train_same, train_different = parse_pairs(same_len=1100, train_or_test="train", train_file=train_file,
                                              test_file=test_file, images=images)
    test_same, test_different = parse_pairs(same_len=500, train_or_test="test", train_file=train_file,
                                            test_file=test_file, images=images)
    x1_train, x2_train = read_images([train_same, train_different], crop)
    y_same_train = np.ones((1100, 1), np.float32)
    y_different_train = np.zeros((1100, 1), np.float32)
    y_train = np.concatenate((y_same_train, y_different_train), axis=0)
    x1_train, x2_train, y_train = shuffle(x1_train, x2_train, y_train, random_state=1)

    x1_test, x2_test = read_images([test_same, test_different], crop)
    y_same_test = np.ones((500, 1), np.float32)
    y_different_test = np.zeros((500, 1), np.float32)
    y_test = np.concatenate((y_same_test, y_different_test), axis=0)
    return x1_train, x2_train, y_train, x1_test, x2_test, y_test


"""
Model Section 
"""


def custom_loss(lambda_t):
    """
    The custom_loss according to the reference paper.
    :param lambda_t: the lambda value for loss l2 norm
    :return: the loss function call
    """

    def loss(y_true, y_pred):
        bxentropy = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        bxe_loss = tf.reduce_mean(bxentropy)
        l2_norms = [tf.nn.l2_loss(vector) for vector in tf.compat.v1.trainable_variables()]
        l2_norm = tf.reduce_sum(l2_norms)
        final_cost = bxe_loss + lambda_t * l2_norm
        return final_cost

    return loss


def step_decay(epoch, lr):
    """
    Control the learning rate decay. Drop 0.01 every epoch.
    :param epoch: the epoch num
    :param lr: the current learning rate
    :return: the updated learning rate
    """
    drop = 0.01
    epochs_drop = 1.0
    lrate = lr * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def get_fit_twinsnet(first_train_x, second_train_x, train_y, conv_num, filter_sizes, filter_nums, batch_size, lambda_t,
                     l2_values, batchnorm=False, bce_loss=False, lrs=True, dropout=0.0):
    """
    Get the model parameters and return a complied tensorflow.keras model
    :param first_train_x: Ndarray. The first twin input.
    :param second_train_x: Ndarray. The second twin input.
    :param train_y: Ndarray. The same/different labels.
    :param conv_num: Int. The number of Conv2D+MaxPool2D blocks in the model.
    :param filter_sizes: List of int. The Conv2D filter size for each Conv2D+MaxPool2D block. for example: 2 is 2X2 filter.
    :param filter_nums: List of int. The Conv2D amount of filters for each Conv2D+MaxPool2D block.
    :param batch_size: int. The batch size.
    :param lambda_t: float. The lambda value for loss l2 norm
    :param l2_values: List of floats. The lambda values for weights l2 norm
    :param batchnorm: if to use batchnorm.
    :param bce_loss: if to use binary crossenrtopy as loss.
    :param lrs: if to use LearningRateScheduler
    :param dropout: dropout rate.
    :return: a fitted tensorflow.keras model
    """
    stride = 1
    padding = "valid"  # no padding
    tf.compat.v1.disable_eager_execution()

    first_input = keras.Input(shape=(first_train_x.shape[1], first_train_x.shape[2], 1))
    second_input = keras.Input(shape=(first_train_x.shape[1], first_train_x.shape[2], 1))
    twin = keras.Sequential()
    for i in range(conv_num - 1):
        twin.add(keras.layers.Conv2D(filters=filter_nums[i],
                                     kernel_size=filter_sizes[i],
                                     strides=stride,
                                     padding=padding,
                                     activation='relu',
                                     use_bias=True,
                                     bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=1e-2),
                                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-2),
                                     data_format="channels_last",
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_values[i])))
        if batchnorm:
            twin.add(tf.keras.layers.BatchNormalization())
        twin.add(keras.layers.MaxPool2D(pool_size=2, strides=1, padding='valid'))
    twin.add(keras.layers.Conv2D(filters=filter_nums[conv_num - 1],
                                 kernel_size=filter_sizes[conv_num - 1],
                                 strides=stride,
                                 padding=padding,
                                 activation='relu',
                                 use_bias=True,
                                 bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=1e-2),
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1e-2),
                                 data_format="channels_last",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_values[conv_num - 1])))
    if batchnorm:
        twin.add(tf.keras.layers.BatchNormalization())
    twin.add(keras.layers.Flatten())
    twins_l1_layer = keras.layers.Lambda(lambda tensors: keras.backend.abs(tensors[0] - tensors[1]))
    twin1 = twin(first_input)
    twin2 = twin(second_input)
    distance = twins_l1_layer([twin1, twin2])
    if dropout != 0.0:
        dropout_l = tf.keras.layers.Dropout(dropout)(distance)
        out = keras.layers.Dense(units=1,
                                 activation='sigmoid',
                                 bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=1e-2),
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=2 * 1e-2))(dropout_l)
    else:
        out = keras.layers.Dense(units=1,
                                 activation='sigmoid',
                                 bias_initializer=tf.random_normal_initializer(mean=0.5, stddev=1e-2),
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=2 * 1e-2))(distance)

    complit_net = keras.Model(inputs=[first_input, second_input], outputs=out)
    if bce_loss:
        complit_net.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.SGD(
                lr=0.01,
                momentum=0.5),
            metrics=['accuracy']
        )
    else:
        complit_net.compile(
            loss=custom_loss(lambda_t),
            optimizer=tf.keras.optimizers.SGD(
                lr=0.01,
                momentum=0.5),
            metrics=['accuracy']
        )
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    start_time = time.time()
    if lrs:
        lr_schedule = LearningRateScheduler(step_decay)
        history = complit_net.fit([first_train_x, second_train_x], train_y,
                                  validation_split=0.3,
                                  epochs=200,
                                  batch_size=batch_size,
                                  callbacks=[lr_schedule, es])
    else:
        history = complit_net.fit([first_train_x, second_train_x], train_y,
                                  validation_split=0.3,
                                  epochs=200,
                                  batch_size=batch_size,
                                  callbacks=[es])
    conversion_time = time.time() - start_time
    return complit_net, history, conversion_time


"""
Analysis Section
"""


def get_final_metric(model, history, test_first_x, test_second_x, test_y, batch_size):
    """
    The method returns the final metric results (loss and accuracy) for the train, validation and test.
    :param model: The evaluated model
    :param history: The evaluated model training history
    :param test_first_x: The test first set of images
    :param test_second_x: The test second set of images
    :param test_y: The test labels set
    :param batch_size: The test batch size
    :return: The final metric results (loss and accuracy) for the train, validation and test.
    """
    final_loss_val = history.history['val_loss'][-1]
    final_acc_val = history.history['val_accuracy'][-1]
    final_acc_train = history.history['accuracy'][-1]
    final_loss_train = history.history['loss'][-1]
    test_metric = model.evaluate([test_first_x, test_second_x], test_y, batch_size=batch_size)
    final_loss_test = test_metric[0]
    final_acc_test = test_metric[1]
    return final_loss_train, final_acc_train, final_loss_val, final_acc_val, final_loss_test, final_acc_test


"""
Experiments Section
"""


def parameter_experiment_stage1(x1_train, x2_train, y_train, x1_test, x2_test, y_test):
    """
    run the different experiments in order to pick the best network parameters.
    :param x1_train: first train images
    :param x2_train: second train images
    :param y_train: train labels
    :param x1_test: first test images
    :param x2_test: second test images
    :param y_test: test labels
    :return: save csv with the experiments results.
    """
    conv_num_op = [2, 3, 4, 5]
    filter_sizes_op = [4, 7, 10, 13, 16, 19]
    filter_nums_op = [64, 128, 192, 256]
    batch_size_op = [15]
    lambda_t_op = [0.001, 0.003, 0.005, 0.007, 0.01]
    l2_values_op = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    report = pd.DataFrame(columns=['lambda_t', 'batch_size', 'conv_num', 'filter_sizes', 'filter_nums', 'l2_values',
                                   'conversion_time', 'loss_train', 'acc_train', 'loss_val', 'acc_val',
                                   'loss_test', 'acc_test'])
    report.index.name = 'ID'
    for lambda_t in lambda_t_op:
        for batch_size in batch_size_op:
            for conv_num in conv_num_op:
                filter_sizes = [random.choice(filter_sizes_op) for i in range(conv_num)]
                filter_sizes.sort(reverse=True)
                filter_nums = [random.choice(filter_nums_op) for i in range(conv_num)]
                filter_nums.sort()
                l2_values = [random.choice(l2_values_op) for i in range(conv_num)]
                fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                                        conv_num=conv_num,
                                                                        filter_sizes=filter_sizes,
                                                                        filter_nums=filter_nums,
                                                                        batch_size=batch_size,
                                                                        lambda_t=lambda_t, l2_values=l2_values)
                loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                                 x1_test,
                                                                                                 x2_test, y_test,
                                                                                                 batch_size)
                report = report.append({'lambda_t': lambda_t, 'batch_size': batch_size, 'conv_num': conv_num,
                                        'filter_sizes': filter_sizes, 'filter_nums': filter_nums,
                                        'l2_values': l2_values, 'conversion_time': conversion_time,
                                        'loss_train': loss_train, 'acc_train': acc_train, 'loss_val': loss_val,
                                        'acc_val': acc_val, 'loss_test': loss_test, 'acc_test': acc_test},
                                       ignore_index=True)
    report.to_csv('ExperimentsStage1/All_Results_1.csv')
    return report


def parameters_plot(data: pd.DataFrame, save_path):
    """
    Generates plot for every parameter used in the experiments
    against different metrics (loss, accuracy and conversion time).
    :param data: pandas DataFrame of parameters and metrics.
    :param save_path: directory path for saving the plots in.
    """
    parameters = data.loc[:, "lambda_t":"l2_values"]
    loss_train = data["loss_train"]
    loss_validation = data["loss_val"]
    loss_test = data["loss_test"]
    acc_train = data["acc_train"]
    acc_validation = data["acc_val"]
    acc_test = data["acc_test"]
    max_loss = max(loss_train.to_list() + loss_validation.to_list() + loss_test.to_list())

    for parameter in parameters.loc[:, :"conv_num"]:
        x = data[parameter]
        # conversion_time
        conversion_time = data["conversion_time"]
        plt.style.use('seaborn-darkgrid')
        plt.scatter(x, conversion_time, color='mediumvioletred', alpha=0.5, s=40)
        plt.xlabel(parameter)
        plt.ylabel("Conversion Time")
        plt.xticks(x.unique())
        plt.title("Conversion Time Depends on {}".format(parameter))
        plt.savefig("{}/conversion_time - {}.png".format(save_path, parameter))
        plt.close()
        # loss
        plt.style.use('seaborn-darkgrid')
        plt.scatter(x, loss_train, color='mediumvioletred', alpha=0.5, s=40, marker='X', label="loss_train")
        plt.scatter(x, loss_validation, color='deepskyblue', alpha=0.5, s=40, marker='o', label="loss_validation")
        plt.scatter(x, loss_test, color='darkorange', alpha=0.5, s=40, marker='v', label="loss_test")
        plt.yticks(np.arange(0, max_loss + 10, 50))
        plt.xlabel(parameter)
        plt.ylabel("Loss")
        plt.xticks(x.unique())
        plt.title("Loss Depends on {}".format(parameter))
        plt.legend()
        plt.savefig("{}/loss - {}.png".format(save_path, parameter))
        plt.close()
        # acc
        plt.style.use('seaborn-darkgrid')
        plt.scatter(x, acc_train, color='mediumvioletred', alpha=0.5, s=40, marker='X', label="acc_train")
        plt.scatter(x, acc_validation, color='deepskyblue', alpha=0.5, s=40, marker='o', label="acc_validation")
        plt.scatter(x, acc_test, color='darkorange', alpha=0.5, s=40, marker='v', label="acc_test")
        plt.xlabel(parameter)
        plt.ylabel("Accuracy")
        plt.xticks(x.unique())
        plt.title("Accuracy Depends on {}".format(parameter))
        plt.legend()
        plt.savefig("{}/acc - {}.png".format(save_path, parameter))
        plt.close()

    for parameter in parameters.loc[:, "filter_sizes":]:
        temp_loss_train = []
        temp_loss_validation = []
        temp_loss_test = []
        temp_acc_train = []
        temp_acc_validation = []
        temp_acc_test = []
        temp_x = []
        temp_conversion_time = []
        x = data[parameter]
        conversion_time = data["conversion_time"]
        for index, values in x.iteritems():
            values = values.strip("[]")
            values = values.split(", ")
            temp_x.extend(values)
            temp_conversion_time += [conversion_time[index]] * len(values)
            temp_loss_train += [loss_train[index]] * len(values)
            temp_loss_validation += [loss_validation[index]] * len(values)
            temp_loss_test += [loss_test[index]] * len(values)
            temp_acc_train += [acc_train[index]] * len(values)
            temp_acc_validation += [acc_validation[index]] * len(values)
            temp_acc_test += [acc_test[index]] * len(values)
        # conversion_time
        temp_x = [float(x) for x in temp_x]
        plt.style.use('seaborn-darkgrid')
        plt.scatter(temp_x, temp_conversion_time, color='mediumvioletred', alpha=0.5, s=40)
        plt.xlabel(parameter)
        plt.xticks(sorted(set(temp_x)))
        plt.ylabel("Conversion Time")
        plt.title("Conversion Time Depends on {}".format(parameter))
        plt.savefig("{}/conversion_time - {}.png".format(save_path, parameter))
        plt.close()
        # loss
        plt.style.use('seaborn-darkgrid')
        plt.scatter(temp_x, temp_loss_train, color='mediumvioletred', alpha=0.5, s=40, marker='X', label="loss_train")
        plt.scatter(temp_x, temp_loss_validation, color='deepskyblue', alpha=0.5, s=40, marker='o',
                    label="loss_validation")
        plt.scatter(temp_x, temp_loss_test, color='darkorange', alpha=0.5, s=40, marker='v', label="loss_test")
        plt.yticks(np.arange(0, max_loss + 10, 50))
        plt.xlabel(parameter)
        plt.xticks(sorted(set(temp_x)))
        plt.ylabel("Loss")
        plt.title("Loss Depends on {}".format(parameter))
        plt.legend()
        plt.savefig("{}/loss - {}.png".format(save_path, parameter))
        plt.close()
        # acc
        plt.scatter(temp_x, temp_acc_train, color='mediumvioletred', alpha=0.5, s=40, marker='X', label="acc_train")
        plt.scatter(temp_x, temp_acc_validation, color='deepskyblue', alpha=0.5, s=40, marker='o',
                    label="acc_validation")
        plt.scatter(temp_x, temp_acc_test, color='darkorange', alpha=0.5, s=40, marker='v', label="acc_test")
        plt.xlabel(parameter)
        plt.xticks(sorted(set(temp_x)))
        plt.ylabel("Accuracy")
        plt.title("Accuracy Depends on {}".format(parameter))
        plt.legend()
        plt.savefig("{}/acc - {}.png".format(save_path, parameter))
        plt.close()


def parameter_experiment_stage2(train_file, test_file, images):
    """
    Run the different experiments for improving the accuracy of the model.
    :param train_file: The train configuration file path
    :param test_file: The test configuration file path
    :param images: The main image directory
    """
    report = []
    x1_train, x2_train, y_train, x1_test, x2_test, y_test = data_loader(train_file, test_file, images, crop=True)
    # Experiment 24 - With crop to 170 (no other change)
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7])
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'paper loss',
                            'Learning rate decay': 'V', 'Dropout rate': '0', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 25 - With crop to 170 and with batchnorm
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            batchnorm=True)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': 'V', 'Loss Function': 'paper loss',
                            'Learning rate decay': 'V', 'Dropout rate': '0', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 26 - With crop to 170, with loss bce
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': 'V', 'Dropout rate': '0', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})

    # Experiment 27 - With crop to 170 without lr dacey, bce, lr=0.0001
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 28 - With crop to 170 without lr dacey, bce, lr=0.0001 + dropout 0.2
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False,
                                                            dropout=0.2)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0.2', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 29 - With crop to 170 without lr dacey, bce, lr=0.0001 + dropout 0.3
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False,
                                                            dropout=0.3)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0.3', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 30 - With crop to 170 without lr dacey, bce, lr=0.0001 + dropout 0.4
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False,
                                                            dropout=0.4)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0.4', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 31 - With crop to 170 without lr dacey, bce, lr=0.0001 + dropout 0.5
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False,
                                                            dropout=0.5)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0.5', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    # Experiment 32 - With crop to 170 without lr dacey, bce, lr=0.0001 + dropout 0.6
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False,
                                                            dropout=0.6)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0.6', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})

    # Experiment 33 - With crop to 170 without lr dacey, bce, lr=0.0001 + dropout 0.7
    fitted_net, history, conversion_time = get_fit_twinsnet(x1_train, x2_train, y_train,
                                                            conv_num=5,
                                                            filter_sizes=[16, 13, 13, 7, 4],
                                                            filter_nums=[64, 128, 192, 192, 256],
                                                            batch_size=15,
                                                            lambda_t=0.003,
                                                            l2_values=[0.3, 0.2, 0.05, 0.05, 0.7],
                                                            bce_loss=True,
                                                            lrs=False,
                                                            dropout=0.7)
    loss_train, acc_train, loss_val, acc_val, loss_test, acc_test = get_final_metric(fitted_net, history,
                                                                                     x1_test,
                                                                                     x2_test, y_test,
                                                                                     15)
    report = report.append({'crop to 170X170': 'V', 'Batch norm': '', 'Loss Function': 'binary cross entropy',
                            'Learning rate decay': '', 'Dropout rate': '0.7', 'Conversion time': conversion_time,
                            'Train Loss': loss_train, 'Train accuracy': acc_train, 'Validation Loss': loss_val,
                            'Validation Accuracy': acc_val, 'Test Loss': loss_test, 'Test Accuracy': acc_test})
    report_df = pd.DataFrame(report)
    report_df.index.name = 'ID'
    report_df.index += 24
    report_df.to_csv('ExperimentsStage2/All_Results_2')


def history_plot(history, save_path):
    """
    Updates the history plots of the optimal model.
    :param history: csv file of the model train history.
    :param save_path: path to save the updated plots.
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.style.use('seaborn-darkgrid')
    plt.plot(epochs, loss, color='mediumvioletred', alpha=0.5, label="Train loss")
    plt.plot(epochs, val_loss, color='deepskyblue', alpha=0.5, label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("{}/Loss history.png".format(save_path))
    plt.close()

    plt.style.use('seaborn-darkgrid')
    plt.plot(epochs, acc, color='mediumvioletred', alpha=0.5, label="Train accuracy")
    plt.plot(epochs, val_acc, color='deepskyblue', alpha=0.5, label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("{}/Accuracy history.png".format(save_path))
    plt.close()


"""
Image Classification Section
"""


def good_bad_examples(model, x1_test, x2_test, y_test):
    """
    Finds a pair of images that the model classified correctly as the same person,
    and a pair of images that the model misclassified them as the same person.
    :param model: The fitted optimal model which to be used to find the examples.
    :param x1_test: Numpy array of the first set of images for each pair in the test set.
    :param x2_test: Numpy array of the second set of images for each pair in the test set.
    :param y_test: The labels vector for the test set.
    """
    good = False
    bad = False
    curr_example = 0
    labels = []
    # labels = pd.DataFrame(columns=['Type','Label'])
    while not good or not bad:
        example1 = x1_test[curr_example].reshape(1, x1_test.shape[1], x1_test.shape[2], x1_test.shape[3])
        example2 = x2_test[curr_example].reshape(1, x1_test.shape[1], x1_test.shape[2], x1_test.shape[3])
        y = list(y_test[curr_example])[0]
        predictions = model.predict([example1, example2])
        if round(list(predictions[0])[0]) == y:
            if not good:
                plt.imsave("ExampleClassification/good_image1.jpg",
                           example1.reshape(example1.shape[1], example1.shape[2]), cmap='Greys_r')
                plt.imsave("ExampleClassification/good_image2.jpg",
                           example2.reshape(example2.shape[1], example2.shape[2]), cmap='Greys_r')
                labels.append({'Type': 'Good Classification', 'Label': y})
                good = True
                curr_example = curr_example + 1
        else:
            if not bad:
                plt.imsave("ExampleClassification/bad_image1.jpg",
                           example1.reshape(example1.shape[1], example1.shape[2]), cmap='Greys_r')
                plt.imsave("ExampleClassification/bad_image2.jpg",
                           example2.reshape(example2.shape[1], example2.shape[2]), cmap='Greys_r')
                labels.append({'Type': 'Bad Classification', 'Label': y})
                bad = True
                curr_example = curr_example + 1
    labels = pd.DataFrame(labels)
    labels.to_csv('ExampleClassification/Labels.csv')


def main():
    if len(sys.argv) < 1:
        print("Error: Not enough parameters")
    else:
        if len(sys.argv) > 2:
            operation = str(sys.argv[1])
            train_file = str(sys.argv[2])
            test_file = str(sys.argv[3])
            images = str(sys.argv[4])
        else:
            operation = str(sys.argv[1])
            train_file = "Data/pairsDevTrain.txt"
            test_file = "Data/pairsDevTest.txt"
            images = "Data/lfw2"
        x1_train, x2_train, y_train, x1_test, x2_test, y_test = data_loader(train_file, test_file, images, crop=True)
        if operation == "Exp-1":
            data = parameter_experiment_stage1(x1_train, x2_train, y_train, x1_test, x2_test, y_test)
            parameters_plot(data, 'ExperimentsStage1/Plots')
        if operation == "Exp-1-plots":
            data = pd.read_csv('ExperimentsStage1/All_Results_1.csv')
            parameters_plot(data, 'ExperimentsStage1/Plots')
        if operation == "Exp-2":
            parameter_experiment_stage2(train_file, test_file, images)
        if operation == "Optimal-200-plot":
            history = pd.read_csv('OptimalModel/Epoch200/history32.csv')
            history_plot(history, 'OptimalModel/Epoch200')
        if operation == "Example-class":
            model = tf.keras.models.load_model('OptimalModel/Epoch20/model32epoch20.h5')
            good_bad_examples(model, x1_test, x2_test, y_test)
        else:
            print("Error: Input is not a valid operation.")


if __name__ == "__main__":
    main()
