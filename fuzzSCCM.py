import time

import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import glob
import math
import os

from scipy.spatial.distance import cdist
def euclidean_distance_np(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))


def f(x):
    return np.exp(-x ** 2)
def fuzz_diff(x,std, scale):
    u = np.zeros(3)
    if x<=-0.5*std:
        u[0] = 1
    elif -0.5*std<x and x<0:
        u[0] = 1-math.exp(-x**2/(2*0.5*std))
        u[1] = math.exp(-x**2/(2*0.5*std))
    elif 0<=x and x<0.5*std:
        u[1] = math.exp(-x**2/(2*0.5*std))
        u[2] = 1-math.exp(-x**2/(2*0.5*std))
    elif x>=0.5*std:
        u[2] = 1
    return u

def cal_pattern(x):
    if x[0] < 0 and x[1] < 0:
        return 0
    elif x[0] == 0 and x[1] < 0:
        return 1
    elif x[0] > 0 and x[1] < 0:
        return 2
    elif x[0] < 0 and x[1] > 0:
        return 3
    elif x[0] == 0 and x[1] > 0:
        return 4
    elif x[0] > 0 and x[1] > 0:
        return 5
def cosine_similarity(vec1, vec2):
    
    dot_product = np.dot(vec1, vec2)


    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    
    else:
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
import csv
import random
paths = glob.glob(r'your_path/*.set')
for i in range(37,44):
        filepath = paths[i]
        raw = mne.io.read_raw_eeglab(filepath, preload=True)
        name = filepath.split('/')[-1].replace('.set','')
        print(name)
   
        freq_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
        }
        freqq = ['delta', 'theta', 'alpha', 'beta']
      
        filtered_data = {}
        hilbert_transformed_data = {}
        instantaneous_phase_all = []
        for band, (low_freq, high_freq) in freq_bands.items():
           
            filtered_data[band] = raw.copy().filter(l_freq=low_freq, h_freq=high_freq)

     
            band_data = filtered_data[band].get_data()
            # print(band_data.shape)   (30, 136303)


            instantaneous_phase_all.append(band_data)
       
        for fre in range(2,3):

          
            count = 0
            chan_n = 30
            matrix= instantaneous_phase_all[fre]
            
            t_len = 1000
            metrics1 = 0
            metrics2 = 0
            for ti in range(0,30000,t_len):
                start_t = time.time()

                xyt = np.stack([
                    np.real(matrix[:, ti - 2:ti - 2 + t_len]),
                    np.real(matrix[:, ti - 1:ti - 1 + t_len]),
                    np.real(matrix[:, ti:ti + t_len])
                ], axis=-1)
                print(xyt.shape)
                diff1 = matrix[:, ti - 1:ti - 1 + t_len] - matrix[:, ti - 2:ti - 2 + t_len]
                diff2 = matrix[:, ti:ti + t_len] - matrix[:, ti - 1:ti - 1 + t_len]

                list_2 = np.concatenate([diff1.ravel(), diff2.ravel()])
                # print(xyt[0][0][0])
                std2 = np.std(list_2)

                x1 = xyt[:, :, 1] - xyt[:, :, 0]  # shape (chan_n, t_len)
                x2 = xyt[:, :, 2] - xyt[:, :, 1]  # shape (chan_n, t_len

                n1 = np.array([fuzz_diff(val, std2, 0.5) for val in x1.ravel()]).reshape(chan_n, t_len, 3) 
                n2 = np.array([fuzz_diff(val, std2, 0.5) for val in x2.ravel()]).reshape(chan_n, t_len, 3)

            
                patterns = np.array([[[i, j] for j in range(3)] for i in range(3)])  # shape (3,3,2)


                weights = n1[:, :, :, None] * n2[:, :, None, :]  # shape (chan_n, t_len, 3, 3)

                real_pattern = np.sum(weights[..., None] * patterns, axis=(2, 3))

                distance = np.zeros((chan_n, t_len, t_len))
                for j in range(chan_n):
                  
                    dist_mat = cdist(xyt[j], xyt[j], metric='euclidean')
                    distance[j] = dist_mat

                aver_nn = np.zeros((chan_n, chan_n, t_len, 3)) 

                for j in range(chan_n):
                    for c in range(chan_n):
                        for t in range(t_len):
                            sort_l = np.argsort(distance[j][t])
                            index = sort_l[:5]
                            for k in range(1,5):
                                key = index[1]
                                inn = index[k]
                                # NN[i][j][t][k] = xyt[i][j][inn]
                                weight = math.exp(-distance[j][t][inn] / distance[j][t][key])  
                                aver_nn[j][c][t] += weight * xyt[c][inn] 
                nn = aver_nn  # (freq, 30,30, 3000, 3)
                # print(nn.shape)
                signare = np.zeros((chan_n, chan_n, t_len, 2))
                xdiff = np.diff(nn, axis=-1)
                list_1 = xdiff.ravel().tolist()
                std = np.std(list_1)
                pc_matirx = np.zeros((chan_n, chan_n))


                x1 = xdiff[..., 0]  # (chan_n, chan_n, t_len)
                x2 = xdiff[..., 1]

                vfuzz = np.vectorize(lambda v: fuzz_diff(v, std, 0.5), signature='()->(k)')  # k=3
                n1 = vfuzz(x1)  # (chan_n, chan_n, t_len, 3)
                n2 = vfuzz(x2)  # (chan_n, chan_n, t_len, 3)


                patterns = np.array([[[i, j] for j in range(3)] for i in range(3)])  # (3,3,2)


                weights = n1[..., :, None] * n2[..., None, :]  # (chan_n, chan_n, t_len, 3, 3)

                
                hhh = np.sum(weights[..., None] * patterns, axis=(-3, -2))  # (chan_n, chan_n, t_len, 2)

                for k in range(t_len):
                    for j in range(chan_n):
                        for t in range(chan_n):
                            x1 = hhh[j][t][k]
                            x2 = real_pattern[t][k]
                            
                            # print(x1, x2)
                            # corr = cosine_similarity(x1,x2)
                            # print(corr)
                            # pc_matirx[t][j] += corr
                            # no1.extend(x1)
                            # no2.extend(x2)
                            x1 = np.array(x1)
                            x2 = np.array(x2)

                            if np.all(x1 == x2):
                                pc_matirx[t][j] += 1
                            #
                            elif np.std(x1)==0 and np.std(x2)==0:
                                pc_matirx[t][j] += 1
                            elif np.std(x1)==0 and np.std(x2)!=0:
                                pc_matirx[t][j] += 0
                            elif np.std(x2)==0 and np.std(x1)!=0:
                                pc_matirx[t][j] += 0
                            else:
                                corr= np.corrcoef(x1, x2)[0, 1]
                                pc_matirx[t][j] += corr
                newpth = 'you_out_path/' + freqq[fre] + '_' + str(count) + '_' + filepath.split('/')[-1].replace('set', 'npy')
                np.save(newpth, pc_matirx)
                print(count)
                count += 1


