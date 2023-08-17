import math
import numpy as np
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import pickle
import time

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

record_list = wfdb.get_record_list('ecg-fragment-high-risk-label/1.0.0')
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
class_counts = find_class_counts(y_df)
demo_plots(x_df, class_counts)

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

def golden_search(metric, a, b, tol):
    measurement_count = 0
    gr = (math.sqrt(5) + 1) / 2
    original = tf.identity(a)
    c = tf.identity(b - (b - a) / gr)
    d = tf.identity(a + (b - a) / gr)
    while metric(original, (a + b) / 2) > tol:
        measurement_count += 1
        if metric(original, c) < metric(original, d):
            b = tf.identity(d)
        else:
            a = tf.identity(c)
        measurement_count += 2
        c = tf.identity(b - (b - a) / gr)
        d = tf.identity(a + (b - a) / gr)
    return (a + b) / 2, measurement_count

def PGD_generate_golden(model, x_original, y_categ, iterations = 5, epsilon = 10, alpha = 1, tol = 0.1, metric = L2):
    x_adversarial = []
    measurement_value_storage = []
    measurement_count_storage = []
    for i in range(len(y_categ)):
        measurement_count = 0
        print(i)
        adversary = tf.identity(x_original[i])
        #Initial random perturbation
        adversary += tf.random.normal(shape = [len(x_original[i])], stddev = epsilon, dtype=tf.dtypes.float64)
        #Golden line search to clip
        adversary, measurements = golden_search(metric, x_original[i], adversary, tol)
        measurement_count += measurements
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
            adversary, measurements = golden_search(metric, x_original[i], adversary, tol)
            measurement_count += measurements
        x_adversarial.append(adversary)
        measurement_count_storage.append(measurement_count)
        measurement_value_storage.append(metric(x_original[i], adversary))
    #Return perturbed input
    return np.array(x_adversarial), np.array(measurement_value_storage), np.array(measurement_count_storage)

def PGD_generate_golden_demo(model, x_original, y_categ, iterations = 5, epsilon = 10, alpha = 1, tol = 0.1, metric = L2):
    x_adversarial = []
    measurement_value_storage = []
    measurement_count_storage = []
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('PGD Demonstration')
    ax.set_xlabel('Time Units (4ms/unit)')
    ax.set_ylabel('Voltage Difference (Normalized)')
    for i in range(len(y_categ)):
        measurement_count = 0
        print(i)
        adversary = tf.identity(x_original[i])
        #Initial random perturbation
        adversary += tf.random.normal(shape = [len(x_original[i])], stddev = epsilon, dtype=tf.dtypes.float64)
        #Golden line search to clip
        adversary, measurements = golden_search(metric, x_original[i], adversary, tol)
        measurement_count += measurements
        adv, = ax.plot(range(len(x_original[i])), adversary, 'b-', label = 'adversary')
        orig, = ax.plot(range(len(x_original[i])), x_original[i], 'g-', label = 'original')
        ax.legend(loc='upper left', title='Classes', bbox_to_anchor=(1, 1.02))
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
            adversary, measurements = golden_search(metric, x_original[i], adversary, tol)
            measurement_count += measurements
            time.sleep(3)
            adv.set_ydata(adversary)
            fig.canvas.draw()
            fig.canvas.flush_events()
        #Retain completed adversary
        adv.set_color('r')
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(3)
        orig.remove()
        adv.remove()
        x_adversarial.append(adversary)
        measurement_count_storage.append(measurement_count)
        measurement_value_storage.append(metric(x_original[i], adversary))
    #Return perturbed input
    plt.ioff()
    return np.array(x_adversarial), np.array(measurement_value_storage), np.array(measurement_count_storage)

def RPS_generate(model, x_original, y_categ, iterations = 5,
                 N = 10, M = 3, epsilon = 0.5, beta = 0.9, metric = L2):
    x_adversarial = []
    metric_storage = []
    measurement_count_storage = []
    stored_epsilon = epsilon
    max_attempts = 50
    for i in range(len(y_categ)):
        print(i)
        #Initial neighbour generation - only keep if misclassifies in one list, allow fails in the other
        sample_zipped = []
        sample = []
        sample_metrics = []
        no_fail_zipped = []
        correct_prediction = np.argmax(y_categ[i])
        measurement_count = 0
        for _ in range(N):
            attempts = 1
            adversary = np.copy(x_original[i])
            adversary += tf.random.normal(shape=[len(x_original[i])], stddev=epsilon, dtype=tf.dtypes.float64)
            adversary_class = np.argmax(model(tf.expand_dims(adversary, axis=0), training = False), axis = 1)[0]
            success_flag = False

            while (not success_flag and attempts < max_attempts):
                if adversary_class != correct_prediction:
                    #Keep adversary and get its distance if effective
                    distance = metric(x_original[i], adversary)
                    measurement_count += 1
                    sample_zipped.append((distance, adversary))
                    no_fail_zipped.append((distance, adversary))
                    success_flag = True
                else:
                    #Generate new adversary if not effective
                    attempts += 1
                    adversary = np.copy(x_original[i])
                    adversary += tf.random.normal(shape = [len(x_original[i])], stddev = epsilon, dtype=tf.dtypes.float64)
                    adversary_class = np.argmax(model(tf.expand_dims(adversary, axis=0), training=False), axis=1)[0]

            if not success_flag:
                distance = metric(x_original[i], adversary)
                measurement_count += 1
                sample_zipped.append((distance, adversary))

        # Sort neighbours by distance and keep the best M
        sample_zipped = sorted(sample_zipped, key=lambda x: x[0])[:M]
        try:
            no_fail_best = sorted(no_fail_zipped, key=lambda x: x[0])[0]
        except:
            no_fail_best = (10000, x_original[i])

        # Expand on neighbours in a shrinking window around the best points for a number of iterations
        for _ in range(iterations):
            print('iterating epsilon = ' + str(epsilon))
            sample_candidate = []
            no_fail_candidate = []
            for element in sample_zipped:
                previous_adversary = element[1]
                # Generate N effective neighbours
                for _ in range(N):
                    attempts = 1
                    adversary = previous_adversary + tf.random.normal(shape=[len(previous_adversary)],
                                                                      stddev=epsilon,
                                                                      dtype=tf.dtypes.float64)
                    adversary_class = \
                    np.argmax(model(tf.expand_dims(adversary, axis=0), training=False), axis=1)[0]
                    success_flag = False

                    while (not success_flag and attempts < max_attempts):
                        if adversary_class != correct_prediction:
                            # Keep adversary and get its distance if effective
                            distance = metric(x_original[i], adversary)
                            measurement_count += 1
                            sample_candidate.append((distance, adversary))
                            no_fail_candidate.append((distance, adversary))
                            success_flag = True
                        else:
                            # Generate new adversary if not effective
                            attempts += 1
                            adversary = previous_adversary + tf.random.normal(
                                shape=[len(previous_adversary)],
                                stddev=epsilon, dtype=tf.dtypes.float64)
                            adversary_class = \
                            np.argmax(model(tf.expand_dims(adversary, axis=0), training=False), axis=1)[0]

                    if not success_flag:
                        distance = metric(x_original[i], adversary)
                        measurement_count += 1
                        sample_candidate.append((distance, adversary))
            # Moving to the next iteration if not a single adversary found
            if len(no_fail_candidate) < 1:
                continue
            # Sort neighbours by distance and keep the best M of the N*M posibilities
            sample_candidate = sorted(sample_candidate, key=lambda x: x[0])[:M]
            no_fail_candidate = sorted(no_fail_candidate, key=lambda x: x[0])[0]
            # If no improvements at all, just resample; otherwise update best points and use tighter window
            if no_fail_candidate[0] < no_fail_best[0]:
                sample_zipped = sample_candidate.copy()
                no_fail_best = no_fail_candidate
                epsilon = beta * epsilon
                print('epsilon update')
        # Retain completed adversary and reset bracket
        if no_fail_best[0] == 10000:
            adversary = tf.identity(sample_zipped[-1][1])
        else:
            adversary = tf.identity(no_fail_best[1])
        x_adversarial.append(adversary)
        distance = metric(x_original[i], adversary)
        measurement_count += 1
        epsilon = stored_epsilon
        print('final distance ' + str(distance))
        metric_storage.append(distance)
        measurement_count_storage.append(measurement_count)
    # Return perturbed input
    return np.array(x_adversarial), np.array(metric_storage), np.array(measurement_count_storage)

def comb_adversary(model, x_original, adversary, y_categ, metric = L2):
    changes = 0
    previous_changes = 0
    change_vector = []
    for i in range(len(y_categ)):
        print('combing input ' + str(i))
        initial_adversary = tf.identity(adversary[i])
        #Go through every element randomly without replacement
        remaining_elements = list(range(len(adversary[i])))
        while len(remaining_elements) > 0:
            #Pick a position
            j = random.sample(remaining_elements, k=1)[0]
            for reduction in [100, 75, 50, 25]:
                current_element = adversary[i][j]
                adversary[i][j] = adversary[i][j] - reduction * (adversary[i][j] - x_original[i][j]) / 100.0
                #Revert changes if not misclassifying anymore
                if np.argmax(model(tf.expand_dims(adversary[i], axis=0), training = False), axis = 1)[0] == np.argmax(y_categ[i]):
                    adversary[i][j] = current_element
                    continue
                changes += 1
                #Stop searching if succesful (previous condition not met)
                break
            #Remove tried element
            remaining_elements.remove(j)
        change_vector.append(changes - previous_changes)
        previous_changes = changes
    return adversary, change_vector

def get_combing_effectiveness(original, old_distances, new_adversary, metric):
    distance_vector = []
    for i in range(len(old_adversary)):
        if old_distances[i] > 0:
            new_distance = metric(original[i], new_adversary[i])
            distance_vector.append((old_distances[i] - new_distance) / old_distances[i])
        else:
            distance_vector.append(0)
    return np.mean(distance_vector)

def compare_distances(original, model, y_categ, algo, metric_name, succesful = True):
    metric_dict = {'L0': L0, 'L1': L1, 'L2': L2, 'Linf': Linf,
                   'cosine': cosine, 'pearson': pearson, 'DTW': DTW, 'frechet': frechet}
    adversary, metric_vector = load_results(algo, metric_name)
    if succesful:
        #Only keep succesful adversaries
        successes = []
        metrics_successes = []
        for i in range(len(adversary)):
            if np.argmax(model(tf.expand_dims(adversary[i], axis=0), training=False), axis=1)[0] != np.argmax(y_categ[i]):
                successes.append(adversary[i])
                metrics_successes.append(metric_vector[i])
        adversary = successes.copy()
        metric_vector = metrics_successes.copy()
    means_dict = {}
    smaller_count = 0
    for name in metric_dict:
        print(name)
        #Only do things for the other metrics than what is called
        if name == metric_name:
            means_dict[name] = np.mean(metric_vector)
        else:
            input_distances = []
            for i in range(len(adversary)):
                print(i)
                input_distances.append(metric_dict[name](original[i], adversary[i]))
            means_dict[name] = np.mean(input_distances)
            # Count how many distances are smaller than the base one
            if np.mean(input_distances) < np.mean(metric_vector):
                smaller_count += 1
    print(means_dict)
    print(smaller_count)
    return smaller_count

def pickle_results(algo, metric, x_adv, metric_vector, extra = ''):
    adv_filename = algo + '_adversaries_' + metric + extra + '.pickle'
    dist_filename = algo + '_distances_' + metric + extra + '.pickle'
    with open(adv_filename, 'wb') as jar:
        pickle.dump(x_adv, jar, protocol=pickle.HIGHEST_PROTOCOL)
    with open(dist_filename, 'wb') as jar:
        pickle.dump(metric_vector, jar, protocol=pickle.HIGHEST_PROTOCOL)

def load_results(algo, metric):
    adv_filename = algo + '_adversaries_' + metric + '.pickle'
    dist_filename = algo + '_distances_' + metric + '.pickle'
    with open(adv_filename, 'rb') as jar:
        x_adv = pickle.load(jar)
    with open(dist_filename, 'rb') as jar:
        metric_vector = pickle.load(jar)
    return x_adv, metric_vector

def show_adversary(count, model, x_test, y_categ, x_adv_orig, x_adv_combed = []):
    current_count = 0
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for i in range(len(x_adv_orig)):
        if np.argmax(model(tf.expand_dims(x_adv_orig[i], axis=0), training=False), axis=1)[0] == np.argmax(y_categ[i]):
            continue
        #If really an adversary, plot it
        current_count += 1
        color_index = 0
        fig, ax = plt.subplots()
        ax.set_xlabel('Time Units (4ms/unit)')
        ax.set_ylabel('Voltage Difference (Normalized)')
        ax.plot(range(len(x_adv_orig[i])), x_adv_orig[i], color=colors[color_index], label= 'raw_adv', lw = 2)
        color_index += 1
        if len(x_adv_combed) > 0:
            ax.plot(range(len(x_adv_combed[i])), x_adv_combed[i], color=colors[color_index], label='comb_adv', ls = '--', lw = 2)
            color_index += 1
        ax.plot(range(len(x_test[i])), x_test[i], color=colors[color_index], label='original', lw=2)
        ax.legend(loc='upper left', title='Classes', bbox_to_anchor=(1, 1.02))

        #If enough adversaries plotted, stop
        if current_count >= count:
            break


model, _, x_test, y_test = trainCNN(x_np, y_np)
#Parameter control panel
metric_name = 'L2'
metric_method = L2
sample_size = 200
alpha = 0.1
tol = 0.05
#0.005, 0.01, 0.05
epsilon = 0.2
#5, 10, 15 (we like 15)
N = 15
#5, 3, 1
M = 3
extra = 'demo'

run_time = time.time()
x_adv, metric_vector, count_vector = PGD_generate_golden_demo(model, x_test[:sample_size], y_test[:sample_size],
                                                         epsilon = epsilon, alpha= alpha, tol = tol, metric = metric_method)
#x_adv, metric_vector, count_vector = RPS_generate(model, x_test[:sample_size], y_test[:sample_size], N = N, M = M,
#                                                  epsilon = epsilon, metric = metric_method)
#x_adv, metric_vector = load_results('rps', metric_name)
run_time = time.time() - run_time

evaluation_original = model.evaluate(x_test[:sample_size], y_test[:sample_size], verbose = 1)
evaluaiton_adversary = model.evaluate(x_adv, y_test[:sample_size], verbose = 1)

smaller_count = compare_distances(x_test[:sample_size], model,
                                  y_test[:sample_size], 'rps', metric_name, succesful=False)

with open('PGD_outputs', 'a') as file:
#with open('RPS_outputs', 'a') as file:
    file.write(metric_name + ' epsilon=' + str(epsilon) + ' alpha=' + str(alpha) + ' tol=' + str(tol) + '\n')
    #file.write(metric_name + ' epsilon=' + str(epsilon) + ' N=' + str(N) + ' M=' + str(M) + '\n')
    file.write('orig_acc: ')
    file.write(str(evaluation_original[1]) + '\n')
    file.write('adv_acc: ')
    file.write(str(evaluaiton_adversary[1]) + '\n')
    file.write('average measurements per input: ')
    file.write(str(np.mean(count_vector)) + '\n')
    file.write('average time per input: ')
    file.write(str(run_time / sample_size) + '\n')

old_adversary = tf.identity(x_adv)
x_adv, change_vector = comb_adversary(model, x_test[:sample_size], x_adv, y_test[:sample_size])
average_reduction = get_combing_effectiveness(x_test[:sample_size], metric_vector, x_adv, metric_method)

pickle_results('pgd', metric_name, x_adv, metric_vector, extra = extra)

with open('PGD_outputs', 'a') as file:
#with open('RPS_outputs', 'a') as file:
    file.write('average changes per input: ')
    file.write(str(np.mean(change_vector)) + '\n')
    file.write('average relative distance reduction from combing: ')
    file.write(str(average_reduction) + '\n')

#show_adversary(2, model, x_test, y_test, old_adversary, x_adv)
plt.show()
