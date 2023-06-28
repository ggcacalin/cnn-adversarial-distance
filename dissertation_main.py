import math
import numpy as np
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn import metrics

from dissertation_metrics import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

plt.rc('font', size=18)
plt.rcParams['figure.constrained_layout.use'] = True

#Create inputs & classes dataframes
'''
record_list = wfdb.get_record_list('ecg-fragment-high-risk-label/1.0.0')
print(record_list)
x_df = wfdb.rdrecord('data\\' + record_list[0], physical=False).to_dataframe()
x_df.rename(columns={'col 1': '0'}, inplace = True)
y_df = pd.DataFrame([int(record_list[0][0])])
for i in range(1, len(record_list)):
    new_x = wfdb.rdrecord('data\\' + record_list[i], physical=False).to_dataframe()
    info = new_x['col 1'].to_list()
    x_df[str(i)] = info
    y_df[str(i)] = [int(record_list[i][0])]
x_df.to_csv('input_dataframe.csv', index=False)
y_df.to_csv('classes_dataframe.csv', index=False)
print(x_df)
print(y_df)
'''
#Read processed dataframes
x_df = pd.read_csv('input_dataframe.csv')
y_df = pd.read_csv('classes_dataframe.csv')
x_np = np.transpose(x_df.to_numpy(dtype='int'))
y_np = y_df.to_numpy(dtype='int')[0]

def find_class_counts(y_np):
    counts = [1,0,0,0,0,0]
    prev = 1
    for i in range(1, len(y_np)):
        if y_np[i] != y_np[i-1]:
            prev = y_np[i]
        counts[prev-1] += 1
    return counts

def demo_plots(x_df, counts):
    #Isolate series
    demo_series = []
    demo_series.append(x_df['0'].to_numpy())
    demo_series.append(x_df['337'].to_numpy())
    demo_series.append(x_df[str(337+72)].to_numpy())
    demo_series.append(x_df[str(337+72+169)].to_numpy())
    demo_series.append(x_df[str(337+72+169+132)].to_numpy())
    demo_series.append(x_df[str(337+72+169+132+106)].to_numpy())
    colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta']
    color_index = 0
    fig, ax = plt.subplots()
    ax.set_xlabel('Time Units (4ms/unit)')
    ax.set_ylabel('Voltage Difference')
    for series in demo_series:
        ax.plot(range(len(series)), series, color = colors[color_index], label = str(color_index + 1))
        color_index += 1
    ax.legend(loc = 'upper left', title = 'Classes', bbox_to_anchor = (1, 1.02))
    plt.show()

#Plotting class overview
#class_counts = find_class_counts(y_df)
#demo_plots(x_df, class_counts)

def trainCNN(x_np, y_np):
    #Getting train / test split
    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size = 0.2, stratify = y_np, random_state = 68)
    with tf.device('/device:GPU:1'):
        epochs = 500
        batch_size = 20
        num_classes = 6

        # Normalize training data
        x_train = normalize(x_train)
        x_test = normalize(x_test)
        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train - 1, num_classes)
        y_test = keras.utils.to_categorical(y_test - 1, num_classes)

        use_saved_model = True
        if use_saved_model:
            model = keras.models.load_model("ECG.model")
        else:
            model = keras.Sequential()
            model.add(Conv1D(filters = 64, kernel_size = 8,
                             input_shape = (x_train.shape[1], 1)))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=64, kernel_size=5))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=64, kernel_size=3))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=8))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=5))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=3))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=8))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=5))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(Conv1D(filters=128, kernel_size=3))
            model.add(BatchNormalization())
            model.add(Activation(activation='relu'))
            model.add(GlobalAveragePooling1D())
            model.add(Dense(num_classes, activation = 'softmax'))

            # Optimiser takes in given hyperparameters
            model.compile(loss = "categorical_crossentropy",
                          optimizer = 'adam', metrics = ["accuracy"])
            model.fit(x_train, y_train, batch_size = batch_size,
                                epochs = epochs, validation_split=0.1)
            model.save("ECG.model.v2")

        # Predictions and model evaluation
        preds = model.predict(x_test)
        y_pred = np.argmax(preds, axis = 1)

        results = model.evaluate(x_test, y_test, batch_size=batch_size)
        eval_acc = results[1]
        y_test_display = np.argmax(y_test, axis = 1)
        confusion = metrics.confusion_matrix(y_test_display, y_pred)
        metrics.ConfusionMatrixDisplay(confusion_matrix = confusion, display_labels = range(1, 7)).plot()

        return model, eval_acc, x_test, y_test

def PGD_generate_simple(model, x_original, y_categ, iterations = 5, epsilon = 10, alpha = 1, tol = 0.1, metric = L2):
    x_adversarial = []
    for i in range(len(y_categ)):
        print(i)
        adversary = tf.identity(x_original[i])
        #Initial random perturbation
        adversary += tf.random.uniform(shape = [len(x_original[i])], minval = -epsilon,
                                       maxval = epsilon, dtype=tf.dtypes.float64)
        #Clip starting point
        while metric(x_original[i], adversary) > epsilon:
            adversary = np.mean([adversary, x_original[i]], axis=0)
            print(adversary[:5])

        for iter in range(iterations):
            print('---------------------------')
            var_adv = tf.expand_dims(adversary, axis=0)
            with tf.GradientTape() as g:
                g.watch(var_adv)
                prediction = model(var_adv, training = False)
                loss = tf.keras.losses.CategoricalCrossentropy()(tf.expand_dims(y_categ[i], axis=0), prediction)
                gradient = g.gradient(loss, var_adv)
            #Perturb and clip
            adversary += alpha * tf.sign(gradient[0])
            while metric(x_original[i], adversary) > epsilon:
                adversary = np.mean([adversary, x_original[i]], axis=0)
                print(adversary[:5])
        #Retain completed adversary
        print(metric(x_original[i], adversary))
        x_adversarial.append(adversary)
    #Return perturbed input
    return np.array(x_adversarial)

def golden_search(metric, a, b, tol):
    gr = (math.sqrt(5) + 1) / 2
    original = tf.identity(a)
    c = tf.identity(b - (b - a) / gr)
    d = tf.identity(a + (b - a) / gr)
    while metric(original, (a + b) / 2) > tol:
        if metric(original, c) < metric(original, d):
            b = tf.identity(d)
        else:
            a = tf.identity(c)
        c = tf.identity(b - (b - a) / gr)
        d = tf.identity(a + (b - a) / gr)
    return (a + b) / 2

def PGD_generate_golden(model, x_original, y_categ, iterations = 5, epsilon = 10, alpha = 1, tol = 0.1, metric = L2):
    x_adversarial = []
    for i in range(len(y_categ)):
        print(i)
        adversary = tf.identity(x_original[i])
        #Initial random perturbation
        adversary += tf.random.uniform(shape = [len(x_original[i])], minval = -epsilon,
                                       maxval = epsilon, dtype=tf.dtypes.float64)
        #Golden line search to clip
        adversary = golden_search(metric, x_original[i], adversary, tol)
        print('-----------------')

        for iter in range(iterations):
            var_adv = tf.expand_dims(adversary, axis=0)
            with tf.GradientTape() as g:
                g.watch(var_adv)
                prediction = model(var_adv, training = False)
                loss = tf.keras.losses.CategoricalCrossentropy()(tf.expand_dims(y_categ[i], axis=0), prediction)
                gradient = g.gradient(loss, var_adv)
            #Perturb and clip
            adversary += alpha * tf.sign(gradient[0])
            adversary = golden_search(metric, x_original[i], adversary, tol)
        #Retain completed adversary
        x_adversarial.append(adversary)
        print(metric(x_original[i], adversary))
    #Return perturbed input
    return np.array(x_adversarial)

def RPS_generate(model, x_original, y_categ, iterations = 5, N = 5, M = 2, epsilon = 1, beta = 0.8, metric = L2):
    x_adversarial = []
    stored_epsilon = epsilon
    max_attempts = N ** 2
    for i in range(len(y_categ)):
        print(i)
        #Initial neighbour generation - only keep if misclassifies
        sample = []
        sample_metrics = []
        correct_prediction = np.argmax(y_categ[i])
        for _ in range(N):
            attempts = 1
            #Randomly generate positions to be perturbed
            perturbed_positions = random.sample(range(len(y_categ)), int(len(y_categ) * attempts / max_attempts))
            adversary = np.copy(x_original[i])
            for pos in perturbed_positions:
                adversary[pos] += tf.random.uniform(shape=[1], minval = -epsilon,
                                       maxval = epsilon, dtype=tf.dtypes.float64)

            #adversary = x_original[i] + tf.random.uniform(shape = [len(x_original[i])], minval = -epsilon,
            #                           maxval = epsilon, dtype=tf.dtypes.float64)
            adversary_class = np.argmax(model(tf.expand_dims(adversary, axis=0), training = False), axis = 1)[0]
            success_flag = False

            while (not success_flag and attempts < max_attempts):
                if adversary_class != correct_prediction:
                    #Keep adversary and get its distance if effective
                    sample.append(adversary)
                    sample_metrics.append(metric(x_original[i], adversary))
                    success_flag = True
                else:
                    #Generate new adversary if not effective
                    attempts += 1
                    perturbed_positions = random.sample(range(len(y_categ)),
                                                        int(len(y_categ) * attempts / max_attempts))
                    adversary = np.copy(x_original[i])
                    for pos in perturbed_positions:
                        adversary[pos] += tf.random.uniform(shape=[1], minval=-epsilon,
                                                            maxval=epsilon, dtype=tf.dtypes.float64)
                    adversary_class = np.argmax(model(tf.expand_dims(adversary, axis=0), training=False), axis=1)[0]

            #Failsafe
            if not success_flag:
                continue
        if len(sample_metrics) < 1:
            x_adversarial.append(x_original[i])
            continue
        # Sort neighbours by distance and keep the best M
        zipped = sorted(zip(sample_metrics, sample))
        zipped = zipped[:M]
        #Expand on neighbours in a shrinking window around the best points for a number of iterations
        for _ in range(iterations):
            best_tuple = list(zipped[0])
            #print(best_tuple[0])
            sample = []
            sample_metrics = []
            for element in zipped:
                previous_adversary = list(element)[1]
                #Generate N effective neighbours
                for _ in range(N):
                    attempts = 1
                    adversary = previous_adversary + tf.random.uniform(shape=[len(previous_adversary)],
                                                                       minval=-epsilon/20,
                                                                       maxval=epsilon/20, dtype=tf.dtypes.float64)
                    adversary_class = np.argmax(model(tf.expand_dims(adversary, axis=0), training=False), axis=1)[0]
                    success_flag = False
                    while (not success_flag and attempts < max_attempts):
                        if adversary_class != correct_prediction:
                            # Keep adversary and get its distance if effective
                            sample.append(adversary)
                            sample_metrics.append(metric(x_original[i], adversary))
                            success_flag = True
                        else:
                            # Generate new adversary if not effective
                            attempts += 1
                            adversary = previous_adversary + tf.random.uniform(shape=[len(previous_adversary)],
                                                                               minval=-epsilon/20,
                                                                               maxval=epsilon/20, dtype=tf.dtypes.float64)
                            adversary_class = np.argmax(model(tf.expand_dims(adversary, axis=0), training=False), axis=1)[0]
            #Failsafe
            if len(sample_metrics) < 1:
                continue
            # Sort neighbours by distance and keep the best M of the N*M posibilities
            new_zipped = sorted(zip(sample_metrics, sample))
            new_zipped = new_zipped[:M]
            # If no improvements at all, just resample; otherwise update best points and use tighter window
            if list(new_zipped[0])[0] < best_tuple[0]:
                zipped = new_zipped.copy()
                epsilon = beta * epsilon
        #Retain completed adversary and reset bracket
        adversary = tf.identity(list(zipped[0])[1])
        x_adversarial.append(adversary)
        epsilon = stored_epsilon
        print(metric(x_original[i], adversary))
    # Return perturbed input
    return np.array(x_adversarial)

model, _, x_test, y_test = trainCNN(x_np, y_np)
#x_adv = PGD_generate_simple(model, x_test, y_test, epsilon=0.1, alpha=0.1, metric = frechet)
#x_adv = PGD_generate_golden(model, x_test, y_test, epsilon = 5, alpha=0.5, tol = 0.1, metric = L2)
x_adv = RPS_generate(model, x_test, y_test, iterations=50, N=7, M=2, epsilon = 0.5, beta = 0.8)
x_adv = normalize(x_adv)
model.evaluate(x_test, y_test, verbose = 1)
model.evaluate(x_adv, y_test, verbose = 1)
for i in range(4):
    colors = ['red', 'green', 'blue', 'orange', 'cyan', 'magenta']
    color_index = 0
    fig, ax = plt.subplots()
    ax.set_xlabel('Time Units (4ms/unit)')
    ax.set_ylabel('Voltage Difference (Normalized)')
    ax.plot(range(len(x_test[i])), x_test[i], color = colors[color_index], label = str(color_index + 1))
    color_index += 1
    ax.plot(range(len(x_adv[i])), x_adv[i], color = colors[color_index], label = str(color_index + 1))
    ax.legend(loc = 'upper left', title = 'Classes', bbox_to_anchor = (1, 1.02))
plt.show()
