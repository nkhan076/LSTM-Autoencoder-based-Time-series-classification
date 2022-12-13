import pandas as pd
import numpy as np
from numpy import sin,cos,pi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def init():
    dict_user= {'ashwaq':[1, 'controlled'],
        'haneen':[2, 'controlled'],
        'hind'  :[3, 'controlled'],
        'nada'  :[4, 'controlled'],
        'shrooq':[5, 'controlled'],
        'amita':[6, 'uncontrolled'],
        'aya':[7, 'uncontrolled'],
        'naima':[8, 'uncontrolled'],
        'nisreen':[9, 'uncontrolled'],
        'sara':[10, 'uncontrolled']

       }
    return dict_user


def R_x(x):
    # body frame rotation about x axis
    return np.array([[1,      0,       0],
                     [0,cos(-x),-sin(-x)],
                     [0,sin(-x), cos(-x)]])
def R_y(y):
    # body frame rotation about y axis
    return np.array([[cos(-y),0,-sin(-y)],
                    [0,      1,        0],
                    [sin(-y), 0, cos(-y)]])
def R_z(z):
    # body frame rotation about z axis
    return np.array([[cos(-z),-sin(-z),0],
                     [sin(-z), cos(-z),0],
                     [0,      0,       1]])

def check_user_ids(user_ids):
#     ids=[]
    unique_values, indice = np.unique(user_ids, return_index=True)
    return unique_values, indice
     
def check_labels(labels):
    unique_values, indice = np.unique(labels, return_index=True)
    return unique_values, indice    

def generate_datasets_for_training(data, user_ids, cat_labels, labels, window_length, scale=True, scaler_type=StandardScaler):
    _l = len(data) 
    data = scaler_type().fit_transform(data)
    Xs = []
    Ys = []
    Y = []
    Z= []
    U=[]
    for i in range(0, (_l - window_length)):
    # because this is an autoencoder - our Ys are the same as our Xs. No need to pull the next sequence of values
        user_unique_values, user_indices = check_user_ids(user_ids[i:i+window_length].values)
        label_unique_values, label_indices = check_labels(labels[i:i+window_length].values)
        if (len(user_unique_values)==1) and (len(label_unique_values)==1):
    #         print(f'{i}-- {unique_values}--{indices}')        
            Xs.append(data[i:i+window_length])
            Ys.append(cat_labels[i:i+window_length])
            Y.append(labels[i:i+window_length])
            Z.append(cat_labels[i])
            U.append(user_ids[i:i+window_length])
    tr_x, ts_x, tr_y, ts_y = [np.array(x) for x in train_test_split(Xs, Z)]
    tr_x, tv_x, tr_y, tv_y = [np.array(x) for x in train_test_split(tr_x, tr_y)]
#     assert tr_x.shape[2] == ts_x.shape[2] == (data.shape[1] if (type(data) == np.ndarray) else len(data))
    return  (tr_x.shape[2], tr_x, tr_y, tv_x, tv_y, ts_x, ts_y, Xs, Ys, Y, Z, U)

