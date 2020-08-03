import os
import random
import logging
import numpy as np
import matplotlib.pyplot as plt 
from keras.utils import to_categorical


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_folds(ex):
    numlist = list(range(4))
    numlist.remove(ex)
    numlist = [str(num) for num in numlist]
    fold = "fold" + "".join(numlist)
    test_fold = "fold" + str(ex)
    return fold, test_fold

def normalization_processing(data):
    
    data_mean = data.mean()
    data_var = data.std()
    
    data = data - data_mean
    data = data / data_var
    
    return data

def signal_processing(data):
    
    data_list = np.zeros((9000), dtype = np.float32)
    a = data.shape[0]

    if a > 9000:
        ran_b = np.random.randint(4500,a-4500)
        data_list = data[ran_b-4500:ran_b+4500] 
    
    if a == 9000:
        data_list = data
    
    if 4500 <= a < 9000:
        data_list[0:a] = data
        data_list[a:9000] = data[0:9000-a]
        
    if 3000 <= a < 4500:
        data_list[0:a] = data
        data_list[a:2*a] = data
        data_list[2*a:9000] = data[0:9000-2*a]        

    if  a < 3000:
        data_list[0:a] = data
        data_list[a:2*a] = data
        data_list[2*a:3*a] = data
        data_list[3*a:9000] = data[0:9000-3*a]  
        
    return data_list

class Generaor():

    def __init__(self, fold, batch_size, category=None):
        self.fold = fold
        self.category = category
        self.batch_size = batch_size

        if self.category is not None:
            self.data_num = len(os.listdir("folds/" + os.path.join(self.fold, self.category) + "/data/"))
        else:
            self.data_num = len(os.listdir("folds/" + self.fold + "/data/"))

    def get_data(self):
        data_list = np.zeros((self.batch_size, 9000, 1), dtype = np.float32)
        lab_list = np.zeros((self.batch_size))
        flag = 0
        
        while True:
            
            list = random.sample(range(self.data_num), self.data_num)
            
            for id in list:
                
                num_id = str(id)
                
                if self.category is not None:
                    data_1 = np.load("folds/"  + os.path.join(self.fold, self.category) + "/data/" + num_id + ".npy")
                    lab_1 = np.load("folds/" + os.path.join(self.fold, self.category) + "/label/" + num_id + ".npy")
                else:
                    data_1 = np.load("folds/" + self.fold + "/data/" + num_id + ".npy")
                    lab_1 = np.load("folds/" + self.fold + "/label/" + num_id + ".npy")

                data = signal_processing(data_1)
                data = normalization_processing(data)
                
                data_list[flag , :, 0] = data
                lab_list[flag] = lab_1
                flag += 1               
                    
                if flag >= self.batch_size:
                    hot_lab = to_categorical(lab_list, num_classes=3)
                    yield [data_list], [hot_lab]
                    
                    flag = 0
                    data_list = np.zeros((self.batch_size, 9000, 1), dtype = np.float32)
                    lab_list = np.zeros((self.batch_size))

