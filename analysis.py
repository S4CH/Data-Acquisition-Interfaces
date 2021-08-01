import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import model_building as model


# Correlation trace computation as improved by StackOverflow
# O - matrix of observed leakage (i.e. traces)
# P - column of prediction
# returns a correlation trace
def corr_coeff(O, P):
    n = P.size
    DO = O - (np.einsum('ij->j', O, dtype='float64') / np.double(n))
    DP = P - (np.einsum('i->', P, dtype='float64') / np.double(n))
    temp = np.einsum('ij,ij->j', DO, DO)
    temp *= np.einsum('i,i->', DP, DP)
    temp = np.dot(DP, DO) / np.sqrt(temp)
    return temp



def plot_corr(X,y):
    corr = corr_coeff(X,y)
    plt.plot(corr)
    plt.xlabel('Points in time')
    plt.ylabel('Correlation')


def analyse_AES(mypath):
    byte = 0
    measure, plaintext = load_all_data(mypath)
    y = model.aes_sbox_model(plaintext,byte)
    plot_corr(measure,y)


def load_all_data(mypath):
    measure = []
    plaintext = []
    for f in listdir(mypath):
        if isfile(join(mypath, f)):
            measure.append(unPackdata(f))
            text = f[4:20]
            text_int= [ord(t) for t in text] # TODO: how to convert it into int??
            plaintext.append(text_int)
    
    return measure,plaintext

