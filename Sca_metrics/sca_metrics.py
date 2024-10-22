import numpy as np
from numba import njit


@njit
def fast_key_rank(nt, n_ge, probabilities_kg_all_traces, nb_guesses, correct_key, key_ranking_sum, success_rate_sum):
    r = np.random.choice(nt, n_ge, replace=False)   #长为3000，元素为1到3000的一维无重复一维随即数组
    probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
    key_probabilities = np.zeros(nb_guesses) #256长的零矩阵
    kr_count = 0
    for index in range(n_ge):
        key_probabilities += probabilities_kg_all_traces_shuffled[index]
        key_probabilities_sorted = np.argsort(key_probabilities)[::-1]   #元素从大到小排列
        key_ranking_good_key = list(key_probabilities_sorted).index(correct_key) #正确密钥的排名放入
        key_ranking_sum[kr_count] += key_ranking_good_key
        if key_ranking_good_key == 1:
            success_rate_sum[kr_count] += 1
        kr_count += 1


def sca_metrics(model, x_data, n_ge, label_key_guess, correct_key):
    nt = len(x_data)
    key_ranking_sum = np.zeros(n_ge)
    success_rate_sum = np.zeros(n_ge)

    output_probabilities = np.log(model.predict(x_data) + 1e-36)

    nb_guesses = len(label_key_guess)
    probabilities_kg_all_traces = np.zeros((nt, nb_guesses))
    for index in range(nt):
        probabilities_kg_all_traces[index] = output_probabilities[index][
            np.asarray([int(leakage[index]) for leakage in label_key_guess[:]])  # array with 256 leakage values (1 per key guess)
        ]
    #probabilities_kg_all_traces为[3000,256]的矩阵,第i行表示第i条曲线，密钥为1到256的可能性
    for run in range(100):
        fast_key_rank(nt, n_ge, probabilities_kg_all_traces, nb_guesses, correct_key, key_ranking_sum, success_rate_sum)

    guessing_entropy = key_ranking_sum / 100
    success_rate = success_rate_sum / 100
    if guessing_entropy[n_ge - 1] < 1:
        result_number_of_traces_ge_1 = n_ge - np.argmax(guessing_entropy[::-1] > 1)
    else:
        result_number_of_traces_ge_1 = n_ge

    return guessing_entropy, success_rate, result_number_of_traces_ge_1
