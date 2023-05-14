
#/ M     -> matrix of memory cells
#// ts    -> suppression threshold
#// N     -> clone number multiplier
#// Nc    -> no. of clones to be generated
#// beta  -> decay of the inverse exponential function
#// gen   -> maximum number of generations


import random
import numpy as np
import CNN 
#import kerassample as ks
import tensorflow as tf
from tensorflow import keras
#import keras as ks
#from keras import metrics

#import LinearRegression as LRS
#import suppress
#from keras import backend as Ks
#from keras import metrics
import clone_mut_selection 
from sklearn.preprocessing import scale
import pdb
import pandas as pd
from numpy import loadtxt
import matplotlib.image as mpimg
from PIL import Image
ts = 0.1
N = 20
Nc = 20
beta = 0.05
gen =5
#10
#A = pd.read_csv('C:/MRI 2019/brca_metabric/data_CNA_transposed.csv', delimiter=',')
#A = pd.read_csv('data.csv', delimiter=',')
#dataset=A.to_numpy()
#size=dataset.shape
#dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
## split into input (X) and output (y) variables
#X = dataset[:,0:8724]
#y = dataset[:,8724]
 
#dataset = loadtxt('Hotel-Bow.csv', delimiter=',')
  
#% Parameters for Ploting the Affinity Landscape
#%vmin = 1; vmax = 0.9;ymin = -10; ymax = 10;
import pywt
import pywt.data
import nibabel as nib
import glob
from skimage.transform import rescale
image_list = []
cols=['class','data']
#df=pd.DataFrame(columns=cols)
arr=np.array([])
list=[]
i=1
def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')
#for filename in glob.glob('C:/Users/DELL/OneDrive/covid/COVID-CT-master/COVID-CT-master/Images-processed/CT_COVID/CT_COVID/*.png'): #assuming gif
for filename in glob.glob('C:/Users/DELL/OneDrive/covid/ITK-SNAP/ITK-SNAP/CT-COVID/*.nii.gz'): 

   #im=Image.open(filename)
    #image_list.append(im)
    
    #try:
        image =nib.load(filename)
        data = image.get_fdata()
        res = rescale(data, 0.250, anti_aliasing=False)
        input_volume=res
        vol_max = np.max(input_volume)
        vol_min = np.min(input_volume)
        res = (input_volume-vol_min)/(vol_max - vol_min)
        #width, height, depth= input_unit.shape
        LBand1D=res.flatten()
        #LBand1D=data.flatten()
        #im=Image.open(filename)
        #coeffs = pywt.dwtn(im, 'db4')
        #LBand=coeffs['aaa']
        #LBand1DU=np.unique(LBand1D)
        #LBand1DU=-np.sort(-LBand1DU)
        #LBand1DU=LBand1DU[0:3000]
        #LBand1DU=padarray(LBand1D,3490)
        LBand1DU=np.insert(LBand1D, 0, 1, axis=0)
        list.append(LBand1DU)
        #arr=np.append(arr,LBand1DU, axis=0)
    #except Exception:
        #pass
#for filename in glob.glob('C:/Users/DELL/OneDrive/covid/COVID-CT-master/COVID-CT-master/Images-processed/CT_NonCOVID/CT_NonCOVID/*.png'): #assuming gif
for filename in glob.glob('C:/Users/DELL/OneDrive/covid/ITK-SNAP/ITK-SNAP/CT-NonCOVID/*.nii.gz'):
    #im=Image.open(filename)
    #image_list.append(im)
    
    try:
        image =nib.load(filename)
        data = image.get_fdata()
        res = rescale(data, 0.25, multichannel=True,anti_aliasing=False)
        input_volume=res
        vol_max = np.max(input_volume)
        vol_min = np.min(input_volume)
        res = (input_volume-vol_min)/(vol_max - vol_min)
        #LBand1D=data.flatten()
        #im=Image.open(filename)
        #coeffs = pywt.dwtn(im, 'db4')
        #LBand=coeffs['aaa']
        LBand1D=res.flatten()
        #print(LBand1D);
        #LBand1DU=np.unique(LBand1D)
        #LBand1DU=-np.sort(-LBand1DU)
        #LBand1DU=LBand1DU[0:3000]
        #LBand1DU=LBand1D
        LBand1DU=np.insert(LBand1D, 0, 0, axis=0)
        list.append(LBand1DU)
    except Exception:
        pass
arr=np.array(list)
max_len = np.max([len(a) for a in arr])
np.random.shuffle(arr)

dataset=np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in arr])
size=max_len  #   -1000
print(size)
vmin = 1; vmax = size-1 #size[0] #round(size[1]/100)
 #;%ymin = 0.25; ymax = 4; #%22 attributes makes 2097152 max binary alternate

#% Initial Random Population Within the Intervals (xmin/xmax; ymin/ymax)
xmin = vmin;
ymin = vmin;
xmax = vmax;
ymax = vmax;

 # % x = Ab(:,1); y = Ab(:,2);
#pdb.set_trace()
fit=np.array([]);
Ab= np.random.choice(vmax, N)
#   %Ab= de2bi(Ab);
#   %Ab= randi([0 1],N,8724);
#  % disp(Ab);
#   % define f here
#   % fit = f(Ab);
fit = CNN.CNNaccfunc(Ab, dataset, Label, size)
#fit=ks.kerasaccfunc(Ab, dataset, size)
#fit= LRS.LR(Ab,dataset, size)
#fit= ks.metrics.binary_accuracy(Ab, dataset, size)
#fit = scale( fit, axis=0, with_mean=True, with_std=True, copy=True )
#   fit = (fitnessfunction(Ab,N));

#%    disp('Press any key to continue...'); pause;   
it = 1; Nold = N + 1; Nsup = N;
FLAG = 0; FLAGERROR = 0;
avfitold = np.mean(fit); 
avfit = avfitold-1;
I = np.max(fit);
vout = []; vavfit = []; vN = np.array([]);
   
 #  % avfitold = mean(fit); avfit = avfitold-1;
  # % vout = []; vavfit = []; vN = [];

# Main Loop 
while (it < gen) and (FLAG == 0) :
    print("going into colone....")
    Ab = clone_mut_selection.clone_mut_selection(Ab,Nc,beta,fit,xmin,xmax, dataset, Label, size)
    print("going into colone....")
#% Immune Network Interactions After a Number of Iterations
    if it%5 == 0: 
        if abs(1-avfitold/avfit) < .002 :
            Ab = suppress(Ab,ts); 
            FLAGERROR = 1; 
            Nsupold = Nsup; Nsup = Ab.size; vN = np.append(vN,Nsup); 
         #% Convergence Criterion 
            if (Nsupold-Nsup) == 0: #% & rem(it,20) == 0, 
                 FLAG = 1; FLAGERROR = 0;           
 #  % Insert randomly generated individuals `
    if (FLAGERROR==1) :
      d = round(.4*N); 
      Ab1 = np.random.choice(8725, d) 
      Ab = np.append(Ab, Ab1); 
      FLAGERROR = 0; 
    Ab=Ab.astype(int)
    print(Ab)
    fit = CNN.CNNaccfunc(Ab, dataset, Label, size)
    #fit = ks.kerasaccfunc(Ab,dataset, size)
    avfitold = avfit; 
    out = np.max(fit);  
    I=np.argmax(fit);
    avfit = np.mean(fit);    
   #% Ploting Results    
   #imprime(1,vxp,vyp,vzp,x,y,fit,it,10); 
    N = Ab.size; 
#% d=0; 
#%     for i=1:1:N-1 
#%       for j=i+1:1:N 
#%           d=d+sqrt((x(i)-x(j))^2+(y(i)-y(j))^2); 
#%       end 
#%     end   
#%        s(it)=2*d/(N*(N-1)); 
    it = it + 1; vout = np.append(vout,out); vavfit = np.append(vavfit,avfit); # vN = [vN,N]; 
    print('It: %d	Max: %f	Av: %f	Net size: %d\n' % (it,out,avfit,N)); 
    wait = input("PRESS ENTER TO CONTINUE.")
    print(Ab)
 #%figure(1); plot(-vout); hold on; plot(-vavfit,'-.'); title('Fitness'); hold off;  
#   figure(1); plot(vout);  plot(vavfit,'.'); title('Fitness'); 
#%figure(2); plot(s); hold on;  
#%figure(3);plot(log(-(-186.9307+vout)));hold on;  
#%figure(4); plot(out); hold on; plot(avfit); title('maximum and average accuracy'); hold off; 
 
