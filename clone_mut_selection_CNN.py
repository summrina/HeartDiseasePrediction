# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:43:46 2019

@author: DELL
"""
import math
import numpy as np
import tensorflow as tf
'''from tensorflow import keras
import keras as ks'''
import kerassample as ks
import BLSTM
#import LinearRegression as LRS
def clone_mut_selection_CNN(Ab,Nc,beta,fitin,xmin,xmax, dataset, size):
# C   -> matrix of clones 
# g   -> vector with Gaussian mutation 
# Ab  -> matrix of antibodies 
# N   -> cardinality of Ab 
# Nc  -> number of clones for each candidate 
    N = Ab.size; 
    c = np.array([]);
    C = np.array([]);
    # if N>numel(fitin)
    #     error('fitin is not big enough')
    # end
    for i in Ab:
       vones = np.ones(Nc); 
       Cc = vones * i; 
       g = (np.random.randn(N)/beta) * math.exp(-beta); 
       #g = (randn(N,L)./beta) .* exp(-fitin(i)); 
       g[0] = 0;#np.zeros(L);	% Keep one previous individual for each clone unmutated 
       c = Cc + g; 
       #% Keeps all elements of the population within the allowed bounds 
       c[c<xmin]=xmin; 
       c[c> xmax]=xmax; 
       c=c.astype(int)   
      # fit= LRS.LR(c, dataset, size)
       fit = CNN.CNNaccfunc(Ab, dataset, size) 
       #fit = BLSTM.BLSTM(Ab, dataset, size) 
       #fit= ks.kerasaccfunc(c, dataset, Label, size) 
       result = np.argmax(fit)
       C = np.append(C, c[result]);
       #ntains only the best individuals of each clone 
    return C
