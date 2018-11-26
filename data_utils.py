from sklearn.model_selection import train_test_split
import numpy as np
import arff

def load_data_set_from_arff(filename):
    file = arff.load(open(filename, 'r'))
    dataset = file['data']
    x = np.array([data[1] for data in dataset])
    # y = np.array(0 if data[0] == 'no' else  1 for data in dataset)
    y = np.array([data[0] for data in dataset])
    return x, y

def split_test_train(data, labels, test_prop = 0.2):
   return train_test_split(data, labels, test_size = test_prop, random_state = 42)

def get_yes_ratio(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))['yes']/len(labels)

def get_possible_labels(labels):
    return np.unique(labels)
