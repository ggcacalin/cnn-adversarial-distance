import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font', size=18)
#plt.rcParams['figure.constrained_layout.use'] = True
x_axis = [1, 2, 3]
colors = ['tab:blue', 'tab:orange', 'tab:green']
labels = ['tol', 'alpha', 'stdev']
title = 'PGD'

def plot_params(param_vector, vector_name, labels, title):
    fig, ax = plt.subplots()
    ax.set_xticks([1, 2, 3])
    for i in range(3):
        if i != 2:
            ax.plot(x_axis, param_vector[i], color=colors[i], label=labels[i], lw=2)
        else:
            ax.plot(x_axis, param_vector[i], color=colors[i], label=labels[i], lw=2, ls='--')
    ax.set_xlabel('Parameter Value')
    if vector_name == 'accdrop':
        ax.set_ylabel('Accuracy Drop')
    elif vector_name == 'measurements':
        ax.set_ylabel('Mean Measurements / Input')
    elif vector_name == 'changes':
        ax.set_ylabel('Mean Changes / Input')
    elif vector_name == 'reduction':
        ax.set_ylabel('Relative Distance Reduction')
    ax.set_title(title + ' Parameter Analysis')
    ax.legend(loc='upper left', title='Parameter', bbox_to_anchor=(1, 1.02))

#tol, alpha, stdev (PGD) / stdev, N, M (RPS)
accdrop = [[0.35, 0.55, 0.65], [0.55, 0.35, 0.5], [0.45, 0.35, 0.4]]
measurements = [[177, 123, 88.35], [132, 177, 207], [168, 177, 180]]
changes = [[427.3, 575.15, 619.9], [544.75, 427.3, 485.65], [498.1, 427.3, 448.75]]
reduction = [[0.5, 0.67, 0.73], [0.6, 0.5, 0.55], [0.57, 0.5, 0.53]]

plot_params(accdrop, 'accdrop', labels, title)
plot_params(measurements, 'measurements', labels, title)
plot_params(changes, 'changes', labels, title)
plot_params(reduction, 'reduction', labels, title)

#L0, L1, L2, Linf, Cosine, Pearson, DTW, Frechet
distances = ['L_0', 'L_1', 'L_2', 'L_inf', 'Cosine', 'Pearson', 'DTW', 'Frechet']


def mirrored_bar(pgd_data, rps_data, figure_title, tick_list, pgd_time = [], rps_time = [], use_time = False):
    left_title = 'PGD ' + figure_title
    right_title = 'RPS ' + figure_title
    data = pd.DataFrame(list(zip(pgd_data, rps_data)), index = distances,
                        columns = [left_title, right_title])
    fig, ax = plt.subplots(figsize=(16,8), ncols=2, sharey=True)
    fig.tight_layout()
    ax[0].grid(zorder = 0)
    ax[1].grid(zorder = 0)
    left_bars = ax[0].barh(data.index, data[left_title], height = 0.5, align='center', color='tab:blue', zorder=10)
    ax[0].set_title(left_title, fontsize=18, pad=15, color='tab:blue')
    ax[0].invert_xaxis()
    right_bars = ax[1].barh(data.index, data[right_title], height = 0.5, align='center', color='tab:orange', zorder=10)
    ax[1].set_title(right_title, fontsize=18, pad=15, color='tab:orange')

    ax[0].set(yticks=data.index, yticklabels=data.index)
    ax[0].yaxis.tick_left()
    ax[0].tick_params(axis='y', colors='black')

    ax[0].set_xticks(tick_list)
    ax[1].set_xticks(tick_list)

    for i, v in enumerate(data[left_title]):
        if v > 0:
            ax[0].text(v, i + 0.085 , str(v), color='white', zorder = 20)
        else:
            ax[0].text(v+0.4, i + 0.085, str(v), color='black', zorder=20)
    if use_time:
        for i, v in enumerate(list(zip(pgd_time, data[left_title]))):
            ax[0].text(v[1] + 25, i + 0.085, str(v[0]), color='black', zorder=20)
        for i, v in enumerate(data[right_title]):
            ax[1].text(v - 20, i + 0.085, str(v), color='white', zorder=20)
        for i, v in enumerate(list(zip(rps_time, data[right_title]))):
            ax[1].text(v[1] + 20, i + 0.085, str(v[0]), color = 'black', zorder=20)
    else:
        for i, v in enumerate(data[right_title]):
            if v > 0:
                ax[1].text(v-0.25, i + 0.08, str(v), color='white', zorder=20)
            else:
                ax[1].text(v+0.2, i + 0.08, str(v), color='black', zorder=20)


    plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0, top=0.93, bottom=0.04, left=0.07, right=0.98)

pgd_acc = [87, 43, 73, 60, 60, 66, 37, 69]
rps_acc = [67, 67, 57, 67, 60, 60, 57, 57]

pgd_meas = [47.1, 140, 132, 102.9, 65.4, 66.75, 162.5, 55]
pgd_time = [0.42, 0.67, 0.21, 0.23, 0.46, 0.68, 825, 67]
rps_meas = [241, 241, 162, 241, 241, 241, 241, 241]
rps_time = [88.44, 95.98, 62.25, 90.93, 49.92, 52.47, 989.93, 198.76]

pgd_changes = [92, 53, 76, 67, 67, 75, 47, 75]
rps_changes = [73, 66, 63, 72, 68, 68, 65, 68]

pgd_reduction = [0, 51, 60, 22, 66, 73, 45, 29]
rps_reduction = [0, 63, 54, 28, 65, 65, 62, 37]

pgd_ranks = [0, 7, 5, 4, 4, 4, 7, 2]
rps_ranks = [0, 7, 6, 4, 4, 4, 7, 4]

pgd_ranks_succesful = [0, 6, 0, 0, 0, 0, 5, 0]
rps_ranks_succesful = [0, 6, 2, 0, 0, 0, 6, 0]

figure_title = '% Accuracy Drop'
tick_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mirrored_bar(pgd_acc, rps_acc, figure_title, tick_list)

figure_title = 'Distance Measurements & Time / Input'
tick_list = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
mirrored_bar(pgd_meas, rps_meas, figure_title, tick_list, pgd_time, rps_time, use_time=True)

figure_title = '% Combing Changes / Input'
tick_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mirrored_bar(pgd_changes, rps_changes, figure_title, tick_list)

figure_title = '% Combing Distance Reduction / Input'
tick_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
mirrored_bar(pgd_reduction, rps_reduction, figure_title, tick_list)

figure_title = ' # Distances Outperformed (All)'
tick_list = [1, 2, 3, 4, 5, 6, 7, 8]
mirrored_bar(pgd_ranks, rps_ranks, figure_title, tick_list)

figure_title = ' # Distances Outperformed (Good)'
tick_list = [1, 2, 3, 4, 5, 6, 7, 8]
mirrored_bar(pgd_ranks_succesful, rps_ranks_succesful, figure_title, tick_list)

#stdev, N, M modified
accdrop = [[0.25, 0.35, 0.75], [0.35, 0.35, 0.4], [0.4, 0.35, 0.4]]
measurements = [[161, 161, 161], [81, 161, 401], [61, 161, 261]]
changes = [[355, 425, 705], [426, 425, 458], [459, 425, 457]]
reduction = [[0.44, 0.51, 0.84], [0.51, 0.51, 0.53], [0.54, 0.51, 0.53]]

labels = ['stdev', 'N', 'M']
title = 'RPS'
plot_params(accdrop, 'accdrop', labels, title)
plot_params(measurements, 'measurements', labels, title)
plot_params(changes, 'changes', labels, title)
plot_params(reduction, 'reduction', labels, title)

plt.show()
