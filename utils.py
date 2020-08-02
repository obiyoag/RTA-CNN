import random
import logging
import numpy as np
from keras.utils import to_categorical


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def get_folders(ex):
    numlist = list(range(4))
    numlist.remove(ex)
    numlist = [str(num) for num in numlist]
    folder = "folder" + "".join(numlist)
    test_folder = "folder" + str(ex)
    return folder, test_folder

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

    def __init__(self, folder, batch_size, category=None):
        self.folder = folder
        self.category = category
        self.self.batch_size = self.batch_size

        if self.category is not None:
            self.data_num = len(os.listdir("folds/" + os.join(self.folder, self.category) + "/data/"))
        else:
            self.data_num = len(os.listdir("folds/" + self.folder + "/data/"))

    def get_data(self):
        data_list = np.zeros((self.batch_size, 9000, 1), dtype = np.float32)
        lab_list = np.zeros((self.batch_size))
        flag = 0
        
        while True:
            
            list = random.sample(range(self.data_num), self.data_num)
            
            for id in list:
                
                num_id = str(id)
                
                if self.category is not None:
                    data_1 = np.load("folds/"  + os.join(self.folder, self.category) + "/data/" + num_id + ".npy")
                    lab_1 = np.load("folds/" + os.join(self.folder, self.category) + "/label/" + num_id + ".npy")
                else:
                    data_1 = np.load("folds/" + folder + "/data/" + num_id + ".npy")
                    lab_1 = np.load("folds/" + folder + "/label/" + num_id + ".npy")

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
