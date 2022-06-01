import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
from random import shuffle

categorical_names = {
    'shoppers': ['OperatingSystems', 'Browser', 'Region', 'TrafficType', 'Weekend'],
    'blast_char': ['SeniorCitizen']
    }

seed_dict = {'aps': 1, 'bank': 1, 'blast_char': 3, 'income': 2, 'shoppers': 4, 'shrutime': 2, 'higgs': 4, 'jannis': 4, 'volkert': 3, 'mix': 1}

def data_prep(data_name, datasplit=[.65, .15, .2], subset=False, named_cls=None, length=None):

    X = pd.read_csv('data_processed/' + data_name + '_data.csv')
    y = pd.read_csv('data_processed/' + data_name + '_label.csv')

    # X = pd.read_csv('data_processed/' + data_name + '_data.csv', nrows=100)
    # y = pd.read_csv('data_processed/' + data_name + '_label.csv', nrows=100)

    if subset:
        ipt_cols = X.columns
        nfeats = len(ipt_cols)
        selected_feat_num = np.random.randint(int(nfeats * 0.7), int(nfeats * 0.8))
        selected_feat = sorted(random.sample(range(nfeats), selected_feat_num))

        label_col = y.columns[0]

        cls_num = len(pd.unique(y[label_col]))
        if cls_num > 2 and type(named_cls) != list:
            # selected_cls = pd.Series(y[label_col]).sample(n=2)
            # print(selected_cls)
            selected_cls = np.arange(cls_num)
            np.random.shuffle(selected_cls)
            # print(selected_cls)
            selected_cls = pd.unique(y[label_col])[selected_cls[:2]]
            print(selected_cls)
            index = y[y[label_col].isin(selected_cls)].index

            X = X.iloc[index, :]
            y = y.iloc[index, :]
        elif cls_num > 2 and type(named_cls) == list:
            selected_cls = pd.unique(y[label_col])[named_cls]
            print(selected_cls)
            index = y[y[label_col].isin(selected_cls)].index

            X = X.iloc[index, :]
            y = y.iloc[index, :]

        if length==None:
            permutation = np.arange(X.shape[0])
            np.random.shuffle(permutation)        
            X = X.iloc[permutation[:int(X.shape[0] * 0.7)], selected_feat]
            y = y.iloc[permutation[:int(X.shape[0])], :]
        else:
            X = X.iloc[:length, selected_feat]
            y = y.iloc[:length, :]
        X.reset_index(drop=True, inplace=True)
        y.reset_index(drop=True, inplace=True)

    categorical_indicator = [False] * len(X.columns)
    for i, col in enumerate(X.columns):
        if X[col].dtypes == 'object':
            categorical_indicator[i] = True
        if data_name in categorical_names.keys() and col in categorical_names[data_name]:
            categorical_indicator[i] = True
        if set([0, 1]) == set(X[col].unique()):
            categorical_indicator[i] = True

    categorical_columns = X.columns[list(np.where(np.array(categorical_indicator)==True)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(np.array(categorical_indicator)==True)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))

    for col in categorical_columns:
        X[col] = X[col].astype("object")

    X["Set"] = np.random.choice(["train", "valid", "test"], p = datasplit, size=(X.shape[0],))

    train_indices = X[X.Set=="train"].index
    valid_indices = X[X.Set=="valid"].index
    test_indices = X[X.Set=="test"].index

    X = X.drop(columns=['Set'])
    temp = X.fillna("MissingValue")
    nan_mask = temp.ne("MissingValue").astype(int)
    
    cat_dims = []
    for col in categorical_columns:
    #     X[col] = X[col].cat.add_categories("MissingValue")
        X[col] = X[col].fillna("MissingValue")
        l_enc = LabelEncoder() 
        X[col] = l_enc.fit_transform(X[col].values)
        cat_dims.append(len(l_enc.classes_))
    for col in cont_columns:
    #     X[col].fillna("MissingValue",inplace=True)
        X.fillna(X.loc[train_indices, col].mean(), inplace=True)
    y = y.values
    # if task != 'regression':
    #     l_enc = LabelEncoder() 
    #     y = l_enc.fit_transform(y)
    l_enc = LabelEncoder() 
    y = l_enc.fit_transform(y)
    X_train, y_train = data_split(X,y,nan_mask,train_indices)
    X_valid, y_valid = data_split(X,y,nan_mask,valid_indices)
    X_test, y_test = data_split(X,y,nan_mask,test_indices)

    train_mean, train_std = np.array(X_train['data'][:,con_idxs],dtype=np.float32).mean(0), np.array(X_train['data'][:,con_idxs],dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    # import ipdb; ipdb.set_trace()
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std

def data_split(X,y,nan_mask,indices):
    x_d = {
        'data': X.values[indices],
        'mask': nan_mask.values[indices]
    }
    
    if x_d['data'].shape != x_d['mask'].shape: 
        raise'Shape of data not same as that of nan mask!'
        
    y_d = {
        'data': y[indices].reshape(-1, 1)
    } 
    return x_d, y_d

class DataSetCatCon(Dataset):
    def __init__(self, X, Y, cat_cols, continuous_mean_std=None):
        
        cat_cols = list(cat_cols)
        # X_mask =  X['mask'].copy()
        X = X['data'].copy()
        con_cols = list(set(np.arange(X.shape[1])) - set(cat_cols))
        self.X1 = X[:,cat_cols].copy().astype(np.int64) #categorical columns
        self.X2 = X[:,con_cols].copy().astype(np.float32) #numerical columns
        # self.X1_mask = X_mask[:,cat_cols].copy().astype(np.int64) #categorical columns
        # self.X2_mask = X_mask[:,con_cols].copy().astype(np.int64) #numerical columns
        self.y = Y['data']
        # if task == 'clf':
        #     self.y = Y['data']#.astype(np.float32)
        # else:
        #     self.y = Y['data'].astype(np.float32)
        self.cls = np.zeros_like(self.y,dtype=int)
        self.cls_mask = np.ones_like(self.y,dtype=int)
        if continuous_mean_std is not None:
            mean, std = continuous_mean_std
            self.X2 = (self.X2 - mean) / std

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # X1 has categorical data, X2 has continuous
        return np.concatenate((self.cls[idx], self.X1[idx])), self.X2[idx],self.y[idx]

# def sub_data_prep(data_name, seed, task, datasplit=[.65, .15, .2], num_tasks=3, class_inc=False, length=None, rearrange=False):
def sub_data_prep(opt, datasplit=[.65, .15, .2], length=None):
    data_name = opt.data_name
    seed = opt.dset_seed
    task = opt.dtask
    num_tasks = opt.num_tasks
    class_inc = opt.class_inc
    rearrange = opt.shuffle

    np.random.seed(seed)
    random.seed(seed)

    cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims = [], [], [], [], [], []

    if not class_inc:   
        for i in range(num_tasks):
            if length == None:
                cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(data_name, datasplit, subset=True)
            else:
                cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(data_name, datasplit, subset=True, length=length[i])
            continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)

            print(cat_idxs)
            print(con_idxs)

            cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int)
            cat_dims_group.append(cat_dims)
            con_idxs_group.append(con_idxs)

            train_ds = DataSetCatCon(X_train, y_train, cat_idxs, continuous_mean_std)
            trainloaders.append(DataLoader(train_ds, batch_size=256, shuffle=True,num_workers=4))

            valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, continuous_mean_std)
            validloaders.append(DataLoader(valid_ds, batch_size=256, shuffle=False,num_workers=4))

            test_ds = DataSetCatCon(X_test, y_test, cat_idxs, continuous_mean_std)
            testloaders.append(DataLoader(test_ds, batch_size=256, shuffle=False,num_workers=4))

            y_dims.append(len(np.unique(y_train['data'][:,0])))

    else:
        if data_name == 'volkert':
            selected_cls = np.arange(10)
            np.random.shuffle(selected_cls)
            # print(selected_cls)
            for i in range(5):
                named_cls = list(selected_cls[2 * i: 2 * (i+1)])
                print(named_cls)
                cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std = data_prep(data_name, datasplit, subset=True, named_cls=named_cls)
                continuous_mean_std = np.array([train_mean,train_std]).astype(np.float32)

                print(cat_idxs)
                print(con_idxs)

                cat_dims = np.append(np.array([1]),np.array(cat_dims)).astype(int)
                cat_dims_group.append(cat_dims)
                con_idxs_group.append(con_idxs)

                train_ds = DataSetCatCon(X_train, y_train, cat_idxs,continuous_mean_std)
                trainloaders.append(DataLoader(train_ds, batch_size=256, shuffle=True,num_workers=4))

                valid_ds = DataSetCatCon(X_valid, y_valid, cat_idxs, continuous_mean_std)
                validloaders.append(DataLoader(valid_ds, batch_size=256, shuffle=False,num_workers=4))

                test_ds = DataSetCatCon(X_test, y_test, cat_idxs, continuous_mean_std)
                testloaders.append(DataLoader(test_ds, batch_size=256, shuffle=False,num_workers=4))

                y_dims.append(len(np.unique(y_train['data'][:,0])))
    
    if rearrange:
        order = [1, 2, 0]
        # order = [2, 1, 0]
        cat_dims_group = [cat_dims_group[i] for i in order]
        con_idxs_group = [con_idxs_group[i] for i in order]
        trainloaders = [trainloaders[i] for i in order]
        validloaders = [validloaders[i] for i in order]
        testloaders = [testloaders[i] for i in order]
        y_dims = [y_dims[i] for i in order]
        # shuffle(cat_dims_group), shuffle(con_idxs_group), shuffle(trainloaders), shuffle(validloaders), shuffle(testloaders), shuffle(y_dims)

    return cat_dims_group, con_idxs_group, trainloaders, validloaders, testloaders, y_dims

# def change_order()