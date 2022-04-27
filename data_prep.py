import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

def process_aps():
    train = pd.read_csv('data/AirSat/train.csv', usecols=range(2, 25)).dropna()
    test = pd.read_csv('data/AirSat/test.csv', usecols=range(2, 25)).dropna()

    data = pd.concat([train, test], ignore_index=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X.to_csv('data_processed/aps_data.csv', index=False)
    y.to_csv('data_processed/aps_label.csv', index=False)

def process_data(path, data_name):
    if data_name == 'bank':
        data = pd.read_csv(path, sep=';')
    elif data_name == 'blast_char':
        data = pd.read_csv(path)
        data['TotalCharges'].replace({' ': np.nan}, inplace=True)
        # data['TotalCharges'] = data['TotalCharges'].to_numeric()
    else:
        data = pd.read_csv(path)
    
    dummy_feat = []
    for col in data.columns:
        if len(pd.unique(data[col])) == 1:
            dummy_feat.append(col)
    data.drop(columns=dummy_feat, inplace=True)

    starting_col = 0
    if data_name in  ['blast_char']:
        starting_col = 1
    if data_name == 'shrutime':
        starting_col = 3

    if data_name in ['volkert']:
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
    else:
        X = data.iloc[:, starting_col:-1]
        y = data.iloc[:, -1]
    X.to_csv('data_processed/' + data_name + '_data.csv', index=False)
    y.to_csv('data_processed/' + data_name + '_label.csv', index=False)


if __name__ == '__main__':
    os.makedirs('./data_processed/', exist_ok = True) 
    process_aps()
    process_data('data/BlastChar/BlastChar.csv', 'blast_char')
    process_data('data/bank/bank-full.csv', 'bank')
    process_data('data/income/income.csv', 'income')
    process_data('data/shoppers/shoppers.csv', 'shoppers')
    process_data('data/shrutime/shrutime.csv', 'shturime')
    process_data('data/volkert/volkert.csv', 'volkert')