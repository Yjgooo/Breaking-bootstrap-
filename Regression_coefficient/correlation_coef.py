import numpy as np
import random
from scipy.special import expit
from sklearn.linear_model import LinearRegression, LogisticRegression
import matplotlib.pyplot as plt

#generate n samples (X_i, Y_i) from some specified distribution

#sample = (X_1, \dots, X_n); K:= the number of bootstrap iterations

def sample_generation(n):

    cov = [[1, 0], [0, 1]] 
    samples = np.random.multivariate_normal([0, 0], cov, n)
    samples_list = samples.tolist()
    
    return samples_list


def correlation_coeff(sample):

    x, y = zip(*sample)
    x = list(x)
    y = list(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2))
    std_y = np.sqrt(np.sum((y - mean_y) ** 2))

    return cov / (std_x * std_y)


def bootstrap(sample, K, type):

    if type == "percentile": 

        n = len(sample)

        list_b_sample = []

        for i in range(K):
    
            b_sample = []

            for j in range(n):
                rand = random.randint(0, n-1)
                b_sample.append(sample[rand])

            b_estimate = correlation_coeff(b_sample)

            list_b_sample.append(b_estimate)
        
        #find the quantile of a list

        quantile = np.quantile(list_b_sample, 0.95) 
        
        return quantile

def main(): 

    boostrap_itr = 200
    sample_size = 15
    experiment_time = 100    
    count = 0

    for i in range(experiment_time):

        sample = sample_generation(sample_size)
        #print(sample)
        quantile = bootstrap(sample, boostrap_itr, "percentile")

        upper_conf_band = quantile * (sample_size ** -1/2) 
        
        if  upper_conf_band > 0:
            count += 1

    print(count / experiment_time)

main()





    