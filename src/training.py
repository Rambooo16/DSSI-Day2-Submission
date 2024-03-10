import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
import numpy as np
import argparse

import data_processor

logging.basicConfig(level=logging.INFO)



def run(data_path,model_path):
    logging.info('Processing Data .....')
    df = data_processor.run(data_path)

    x_train = df[:,1:]
    y_train = df[:,0]

    logging.info('Start Training...')
    # Train Classifier
    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1], activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.SGD(learning_rate=0.001))

    #Set class weights for imbalanced dataset
    class_weights = {0: len(y_train)/np.sum(y_train==0),
                    1: len(y_train)/np.sum(y_train==1)}
    
    model.fit(x_train, y_train, epochs=100, class_weight=class_weights, verbose = 0)
    logging.info('Training completed.')

    model.save(model_path+'NN_model.h5')

    return None

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path')
    argparser.add_argument('--model_path')
    args = argparser.parse_args()
    run(args.data_path,args.model_path)