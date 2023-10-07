import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

import pandas as pd
import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle

import os
import pandas as pd
import soundfile as sf
import numpy as np
import math
import scipy.io.wavfile, scipy.signal
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler

from sys import platform
if platform == "linux" or platform == "linux2":
    # linux
    path='/home/vedant/projects/'
elif platform == "darwin":
    # OS X
    path='/Users/vedant/Desktop/Programming/'

cols=['video_id', 'start_time', 'mid_ts', 'label', 'average_zcr',
       'zcr_stddev', 'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean',
       'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean',
       'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean',
       'mfcc13_mean', 'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std',
       'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std',
       'mfcc10_std', 'mfcc11_std', 'mfcc12_std', 'mfcc13_std',
       'delta_mfcc1_mean', 'delta_mfcc2_mean', 'delta_mfcc3_mean',
       'delta_mfcc4_mean', 'delta_mfcc5_mean', 'delta_mfcc6_mean',
       'delta_mfcc7_mean', 'delta_mfcc8_mean', 'delta_mfcc9_mean',
       'delta_mfcc10_mean', 'delta_mfcc11_mean', 'delta_mfcc12_mean',
       'delta_mfcc13_mean', 'delta_mfcc1_std', 'delta_mfcc2_std',
       'delta_mfcc3_std', 'delta_mfcc4_std', 'delta_mfcc5_std',
       'delta_mfcc6_std', 'delta_mfcc7_std', 'delta_mfcc8_std',
       'delta_mfcc9_std', 'delta_mfcc10_std', 'delta_mfcc11_std',
       'delta_mfcc12_std', 'delta_mfcc13_std',
       'centroid_mean','centroid_std',
       'contrast_mean','contrast_std',
       'flatness_mean','flatness_std',
       'rolloff_mean','rolloff_std','rms_mean','rms_std','vggish']
       
d=np.load(path+'ScreamDetection/resources/working_data/vocal_only_features.npy',allow_pickle=True)
df = pd.DataFrame(d,columns=cols)

lut = pd.read_csv(path+'ScreamDetection/resources/dataset/lookup_new.csv')

df.drop('vggish',axis=1,inplace=True)
feature_df=df
mapping=[]
for index,row in feature_df.iterrows():
    if row['label'] == 'clean':
        mapping.append(0)
    if row['label'] == 'highfry':
        mapping.append(1)
    if row['label'] == 'layered':
        mapping.append(1)
    if row['label'] == 'lowfry':
        mapping.append(1)
    if row['label'] == 'midfry':
        mapping.append(1)
    if row['label'] == 'no_vocals':
        mapping.append(2)

feature_df.insert(4,'label_mapped',mapping)

cols=['video_id', 'start_time', 'mid_ts', 'label', 'label_mapped',
       'average_zcr', 'zcr_stddev', 'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean',
       'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean',
       'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean',
       'mfcc13_mean', 'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std',
       'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std',
       'mfcc10_std', 'mfcc11_std', 'mfcc12_std', 'mfcc13_std',
       'delta_mfcc1_mean', 'delta_mfcc2_mean', 'delta_mfcc3_mean',
       'delta_mfcc4_mean', 'delta_mfcc5_mean', 'delta_mfcc6_mean',
       'delta_mfcc7_mean', 'delta_mfcc8_mean', 'delta_mfcc9_mean',
       'delta_mfcc10_mean', 'delta_mfcc11_mean', 'delta_mfcc12_mean',
       'delta_mfcc13_mean', 'delta_mfcc1_std', 'delta_mfcc2_std',
       'delta_mfcc3_std', 'delta_mfcc4_std', 'delta_mfcc5_std',
       'delta_mfcc6_std', 'delta_mfcc7_std', 'delta_mfcc8_std',
       'delta_mfcc9_std', 'delta_mfcc10_std', 'delta_mfcc11_std',
       'delta_mfcc12_std', 'delta_mfcc13_std', 'centroid_mean', 'centroid_std',
       'contrast_mean', 'contrast_std', 'flatness_mean', 'flatness_std',
       'rolloff_mean', 'rolloff_std', 'rms_mean', 'rms_std']
from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy='not minority',random_state=0)
from collections import Counter
X = feature_df.to_numpy()
y=feature_df[['label_mapped']].to_numpy()

X_under, y_under = undersample.fit_resample(X, y)

undersampled_data = pd.DataFrame(X_under,columns=cols)
undersampled_data['label_mapped'] = y_under
#print(undersampled_data)


from sklearn.model_selection import GroupShuffleSplit
train_inds, test_inds = next(GroupShuffleSplit(test_size=.2, n_splits=2, random_state = 42).split(lut, groups=lut['band_name']))

train = lut.iloc[train_inds]
test = lut.iloc[test_inds]

train_ids = train['video_id'].to_numpy()
test_ids = test['video_id'].to_numpy()

#df_final = df
df_final = undersampled_data
train = df_final[df_final.video_id.isin(train_ids)]
test = df_final[df_final.video_id.isin(test_ids)]

features=['average_zcr', 'zcr_stddev', 'mfcc1_mean', 'mfcc2_mean', 'mfcc3_mean',
       'mfcc4_mean', 'mfcc5_mean', 'mfcc6_mean', 'mfcc7_mean', 'mfcc8_mean',
       'mfcc9_mean', 'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean',
       'mfcc13_mean', 'mfcc1_std', 'mfcc2_std', 'mfcc3_std', 'mfcc4_std',
       'mfcc5_std', 'mfcc6_std', 'mfcc7_std', 'mfcc8_std', 'mfcc9_std',
       'mfcc10_std', 'mfcc11_std', 'mfcc12_std', 'mfcc13_std',
       'delta_mfcc1_mean', 'delta_mfcc2_mean', 'delta_mfcc3_mean',
       'delta_mfcc4_mean', 'delta_mfcc5_mean', 'delta_mfcc6_mean',
       'delta_mfcc7_mean', 'delta_mfcc8_mean', 'delta_mfcc9_mean',
       'delta_mfcc10_mean', 'delta_mfcc11_mean', 'delta_mfcc12_mean',
       'delta_mfcc13_mean', 'delta_mfcc1_std', 'delta_mfcc2_std',
       'delta_mfcc3_std', 'delta_mfcc4_std', 'delta_mfcc5_std',
       'delta_mfcc6_std', 'delta_mfcc7_std', 'delta_mfcc8_std',
       'delta_mfcc9_std', 'delta_mfcc10_std', 'delta_mfcc11_std',
       'delta_mfcc12_std', 'delta_mfcc13_std', 'centroid_mean', 'centroid_std',
       'contrast_mean', 'contrast_std', 'flatness_mean', 'flatness_std',
       'rolloff_mean', 'rolloff_std', 'rms_mean', 'rms_std']
x_train = train[features].to_numpy()
y_train_hot = to_categorical(train['label_mapped'].to_numpy())

x_test1 = test[features].to_numpy()
y_test_hot1 = to_categorical(test['label_mapped'].to_numpy())

x_test,X_valid,y_test_hot,y_valid=train_test_split(x_test1,y_test_hot1,test_size=0.2, random_state=42)

X_train=x_train
X_test=x_test

X_train=np.array(X_train)
X_test=np.array(X_test)

scaler = StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)


X_train = X_train.reshape(-1, 64)
X_test = X_test.reshape(-1, 64)

import pandas as pd
def train_models(X_train,y_train_hot,X_test,y_test_hot,epochs,batch_size,lr,layer1_nodes,optimiser,loss):
    model = Sequential()
    input_shape = (64,1)#(128, 87, 1)
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(6, activation='softmax'))
    model.add(Dense(layer1_nodes,input_dim=64, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    if optimiser=='adadelta':
        optim=keras.optimizers.Adadelta(learning_rate=lr)
    if optimiser == 'adam':
        optim=keras.optimizers.Adam(learning_rate=lr)

    if loss == 'crossentropy':
        loss_fn = keras.losses.categorical_crossentropy

    model.compile(loss=loss_fn,
                optimizer=optim,
                metrics=['accuracy'])
    model.build(input_shape)
    model.summary()
    # fit the model
    history=model.fit(X_train, y_train_hot,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_data=(X_test, y_test_hot))
    training_loss=history.history['loss']
    validation_loss=history.history['val_loss']
    training_acc=history.history['accuracy']
    validation_acc=history.history['val_accuracy']
    df=pd.DataFrame()
    df['optimiser'] = optimiser
    df['epochs'] = epochs
    df['learning_rate'] = lr
    df['layer1_nodes'] = layer1_nodes
    df['batch_size'] = batch_size
    df['training_loss'] = training_loss
    df['validation_loss'] = validation_loss
    df['training_acc'] = training_acc
    df['validation_acc'] = validation_acc
    
    lr_str=str(lr).replace('.','_')
    model_name=f'fcnn_optim-{optimiser}_layer1-{layer1_nodes}_batch-{batch_size}_epochs-{epochs}_lr-{lr_str}'
    
    model.save(f'{path}ScreamDetection/CNN/trained_models/fcnn/{model_name}')
    df.to_csv(f'{path}ScreamDetection/CNN/trained_models/fcnn/{model_name}.csv')


if __name__ == '__main__':
    lr_values=[0.001,0.005,0.01,0.05]
    batch_size_values=[256,512,1024]
    layer1_node_values=[8,32,64,256,512,1024,2048]
    epoch_values=[1000,2500,5000,10000]
    optimisers=['adadelta','adam']

    for lr in lr_values:
        for batch_size in batch_size_values:
            for layer1_node in layer1_node_values:
                for epoch in epoch_values:
                    for optimiser in optimisers:
                        lr_str=str(lr).replace('.','_')
                        model_name=f'fcnn_optim-{optimiser}_layer1-{layer1_node}_batch-{batch_size}_epochs-{epoch}_lr-{lr_str}'
                        p=f'{path}ScreamDetection/CNN/trained_models/fcnn/{model_name}'
                        if not os.path.exists(p):
                            print(f"Training model with the following params: optimiser-{optimiser}, lr-{lr}, epoch={epoch}, batch_size-{batch_size}, layer1_nodes={layer1_node}")
                            train_models(X_train,y_train_hot,X_test,y_test_hot,epochs=epoch,batch_size=batch_size,lr=lr,layer1_nodes=layer1_node,optimiser=optimiser,loss='crossentropy')
                        else:
                            print("This model has already been run, moving to the next")