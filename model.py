from tensorflow.keras import metrics
from tensorflow.keras.layers import Input, LSTM, RepeatVector, Dense, TimeDistributed
# import keras
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import save_model


def get_autoencoder(window_length,feats):
    inp_series = Input(shape=(window_length,feats))
    enc1 = LSTM(128, kernel_initializer='he_uniform', batch_input_shape=(None, window_length, feats), return_sequences=True, name='encoder_1')(inp_series)
    enc2 = LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2')(enc1)
    enc3 = LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_3')(enc2)
    enc4 = LSTM(16, kernel_initializer='he_uniform', return_sequences=False, name='encoder_4')(enc3)
    bridge = RepeatVector(window_length, name='encoder_decoder_bridge')(enc4)            
    dec1 = LSTM(16, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1')(bridge)
    dec2 = LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2')(dec1)            
    dec3 = LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3')(dec2)
    dec4 = LSTM(128, kernel_initializer='he_uniform', return_sequences=True, name='decoder_4')(dec3)
    recon_series = TimeDistributed(Dense(feats))(dec4)           

    autoencoder_model = Model(inp_series, recon_series)
    encoder_model = Model(inp_series, enc4)
    print(autoencoder_model.summary())
    #encoder_model.summary()
    return autoencoder_model, encoder_model

def get_classifier(num_classes, enc_layer, autoencoder_model):
#     num_classes = len(data_labels)
    inp_series = autoencoder_model.layers[0].output
    enc4 = autoencoder_model.layers[enc_layer].output
    fc1 = Dense(256, name='fc1',activation='relu')(enc4)
    op = Dense(num_classes, name='output', activation='softmax')(fc1)

    classifier_model = Model(inp_series, op)
    print(classifier_model.summary())
    
    return classifier_model
    
def train_autoencoder(autoencoder_model, train_X, valid_X, model_name, epochs= 10, batch_size= 32):
    autoencoder_model.compile(loss="mse",optimizer='adam')
    ae_history = autoencoder_model.fit(x=train_X, y=train_X, validation_data=(valid_X, valid_X), epochs=epochs, batch_size=batch_size).history
    save_model(autoencoder_model, f"{model_name}_autoencoder_model.h5")
    return ae_history


def train_classfier(classifier_model, train_X, train_Y, valid_X, valid_Y, model_name, epochs=50, batch_size=32):
    classifier_model.compile(loss="categorical_crossentropy",optimizer='adam')
    cl_history = classifier_model.fit(x=train_X, y=train_Y, validation_data=(valid_X, valid_Y), epochs=epochs, batch_size=batch_size).history
    save_model(classifier_model, f"{model_name}_cl_model.h5")
    return cl_history