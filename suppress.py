# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:44:50 2019

@author: DELL
"""

import math
import numpy as np
import kerassample as ks
#import LinearRegression as LRS
import opt_aiNet_fs_H_CNN
#% % Function suppress self-recognizing and non-stimulated Ab from Memory (M)
#% function [M] = suppress(M,ts);
#% % M   -> memory matrix
#% % D1  -> idiotypic affinity matrix
#% D1 = dist(M,M');
#% aux = triu(D1,1);
#% [Is,Js] = find(aux>0 & aux<ts);
#% if ~isempty(Is),
#%    Is = ver_eq(Is);
#%    M = extract(M,Is);
#%    % D1 = extract(D1,Is);
#% end;
#% % D1 = dist(M,M');
# Function Extracts lines from M indexed by I

def suppress(Ab,ts):
#% Given that the pairwise distance between any to Abâ€™s is below ts, stay only with the one with higher fitness
	
#	% M   -> memory matrix 
#% D1  -> idiotypic affinity matrix 
	Iaux = np.array([]);
#%    fit = f(Ab);
for i in range(len(Ab)):
  for j in  range(len(Ab)):
    #!= not so important
    ##earlier question I asked lets me figure that one out
    d[i,j]=Ab[i]-Ab[j];
    aux = np.triu(d, 1);
    aux=abs(aux.astype(int))
    Is = np.argwhere(np.logical_and(aux>0 and aux<ts));
    while Is.size != 0:
        for i in Is:
            #fit1= LRS.LR(c, dataset, i[0])
            #fit2= LRS.LR(c, dataset, i[1])
            fit1=ks.kerasaccfunc(i[0])
            fit2=ks.kerasaccfunc(i[1])
            if fit1 >= fit2: 
                Ab = Ab[Ab!=i[0]];
            else:
                Ab = Ab[Ab!=i[1]]; 
        for i in range(len(Ab)):
          for j in  range(len(Ab)):
        #!= not so important
        ##earlier question I asked lets me figure that one out
            d[i,j]=Ab[i]-Ab[j];
        aux = np.triu(d, 1);
        aux=abs(aux.astype(int))
        Is = np.argwhere(np.logical_and(aux>0 and aux<ts));
        for i in Is:
            #fit = CNN.CNNaccfunc(Ab[0], dataset, size)
            #fit = CNN.CNNaccfunc(Ab[1], dataset, size)
            fit1=ks.kerasaccfunc(i[0])
            fit2=ks.kerasaccfunc(i[1])
            #fit1= LRS.LR(c, dataset, i[0])
            #fit2= LRS.LR(c, dataset, i[1])
            if fit1 >= fit2: 
                Ab = Ab[Ab!=i[0]];
            else:
                Ab = Ab[Ab!=i[1]]; 
   