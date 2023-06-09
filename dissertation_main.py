import numpy as np
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D

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
    ax.legend(loc='upper left', title='Classes', bbox_to_anchor=(1, 1.02))
    plt.show()

#Plotting class overview
#class_counts = find_class_counts(y_df)
#demo_plots(x_df, class_counts)

def trainCNN(x_np, y_np):
    #Getting train / test split
    x_train, x_test, y_train, y_test = train_test_split(x_np, y_np, test_size = 0.2, stratify= y_np, random_state=68)
    with tf.device('/device:GPU:1'):
        epochs = 20
        batch_size = 32
        num_classes = 6

        # Normalize training data
        x_train = normalize(x_train)
        x_test = normalize(x_test)

        # Convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train - 1, num_classes)
        y_test = keras.utils.to_categorical(y_test - 1, num_classes)

        use_saved_model = False
        if use_saved_model:
            model = keras.models.load_model("ECG.model")
        else:
            model = keras.Sequential()
            model.add(Conv1D(filters = 64, kernel_size = 3, input_shape=(x_train.shape[1], 1), activation='relu'))
            model.add(Conv1D(filters = 64, kernel_size=3, activation='relu'))
            model.add(Dropout(0.5))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Flatten())
            model.add(Dense(100, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))

            # Optimiser takes in given hyperparameters
            model.compile(loss="categorical_crossentropy",
                          optimizer='adam', metrics=["accuracy"])
            model.fit(x_train, y_train, batch_size=batch_size,
                                epochs=epochs)
            model.save("ECG.model")

        # Predictions and model evaluation
        preds = model.predict(x_test)
        y_pred = np.argmax(preds, axis=1)

        results = model.evaluate(x_test, y_test, batch_size=batch_size)
        eval_acc = results[1]

        log_loss = metrics.log_loss(y_test, preds)
        return -log_loss, eval_acc

log_loss, eval_acc = trainCNN(x_np, y_np)
print('--------------------------------------------------------------')
print(log_loss)
print(eval_acc)
