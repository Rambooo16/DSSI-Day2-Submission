from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd

import src.data_processor as data_processor


def evaluate():
    model = load_model('models/NN_model.h5')
    df = data_processor.run('data/data_failure_prediction_test.csv')

    x_test = df[:,1:]
    y_test = df[:,0]
    # Evaluate the confusion matrix using the test data

    y_predict_class = (model.predict(x_test) > 0.5).astype("int32")
    result = pd.DataFrame(confusion_matrix(y_test, y_predict_class), index=['true:0', 'true:1'], columns=['pred:0', 'pred:1'])

    return result
