import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer,minmax_scale


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def save_data(data_path,df):
    df.to_csv(data_path.replace('.csv','_processed.csv'),index=False)
    return None

def pre_processed(df):
    df.fillna(0,inplace=True)
    df.replace('na',0,inplace=True)
    df = df.values
    data_preprocess = []
    for i in range(df.shape[1]):
        data_temp = []
        if (i == 0):
            catBinarizer = LabelBinarizer()
            data_temp = catBinarizer.fit_transform(df[:,i])
        else:
            data_temp = minmax_scale(df[:,i].astype(float))
            data_temp = np.reshape(data_temp,(len(data_temp),1))
        if len(data_preprocess) == 0:
            data_preprocess = data_temp
        else:
            data_preprocess = np.hstack([data_preprocess,data_temp])
    
    return data_preprocess

def run(data_path):
    df = load_data(data_path)
    df_processed = pre_processed(df)
    save_data(data_path,pd.DataFrame(df_processed))
    return df_processed

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data_path",type='str')
    args = argparser.parse_args()
    run(args.data_path)