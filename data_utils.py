from sklearn.model_selection import train_test_split
import numpy as np
import arff

def load_data_set_from_arff(filename):
    file = arff.load(open(filename, 'r'))
    dataset = np.array(file['data'])
    x = [data[1] for data in dataset]
    y = [data[0] for data in dataset]
    return x, y

def split_test_train(data, labels, test_prop = 0.2):

   return train_test_split(data, labels, test_size = test_prop, random_state = 42)

