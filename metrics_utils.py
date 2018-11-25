import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_mean(collection):
    return np.mean(np.array(collection))

def calculate_matrix_mean(matrix_collection):
    sum_cm = np.matrix(matrix_collection[0])
    for cm in matrix_collection[1:]:
        sum_cm += np.matrix(cm)
    return sum_cm/len(matrix_collection)

def print_confusion_matrix(cm, labels):
    print("Matriz de confusão: ")
    print("\t\tV\tV")
    print("\t\t{}\t{}".format(labels[0], labels[1]))
    print("P\t{}\t{}\t{}".format(labels[0], cm.item(0), cm.item(1)))
    print("P\t{}\t{}\t{}".format(labels[1], cm.item(2), cm.item(3)))

