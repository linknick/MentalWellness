import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self,CONFIG = None):
        self.use_tensor = False
        if CONFIG['model_type']=='NN':
            self.use_tensor = True
        self.batch_size = CONFIG['batch_size']

    def split(self, DATA, CONFIG, test_size=0.2, random_state=42):
        target = CONFIG['target']
        exclude = CONFIG['exclude']
        DATA.drop(exclude, axis=1, inplace=True)#remove useless column
        #one-hot encoding
        objs = DATA.select_dtypes(include="object").columns
        DATA = pd.get_dummies(DATA, columns=objs)
        train, test = train_test_split(DATA, test_size = test_size, random_state = random_state, shuffle = True)#split bY 8:2
        Y_train, X_train = train[target], train.drop(target, axis=1, inplace = False)
        Y_test, X_test = test[target], test.drop(target, axis=1, inplace = False)
        
        if self.use_tensor:
            X_train = torch.from_numpy(X_train.to_numpy(dtype=np.float32))
            Y_train = torch.from_numpy(Y_train.to_numpy(dtype=np.float32)).view(-1,1)
            X_test = torch.from_numpy(X_test.to_numpy(dtype=np.float32))
            Y_test = torch.from_numpy(Y_test.to_numpy(dtype=np.float32)).view(-1,1)
            
            train_dataset = TensorDataset(X_train, Y_train)
            test_dataset = TensorDataset(X_test, Y_test)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return train_loader, test_loader, X_train.shape[1]
        else:
            return (X_train, Y_train), (X_test, Y_test), X_train.columns.size