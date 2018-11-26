# -*- coding: utf-8 -*-


import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import matplotlib as mpl
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
#
# def plot_one(matrix,path = '.',nome = 'Matriz',save = False,show = True):
# 		"""
# 		:param matrix:
# 		:param path:
# 		:param nome:
# 		:param save:
# 		:param show:
# 		:return:
# 		"""
# 		plt.figure()
# 		matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
# 		for j in range(len(matrix)):
# 			for k in range(len(matrix)):
# 				if ((j != k) and ((matrix[j][k]) != 0)):
# 					matrix[j][k] = (-1 * matrix[j][k])
#
#
#
# 		plt.xticks([i for i in range(0, 5)])
# 		plt.yticks([i for i in range(0, 5)])
# 		plt.title(nome[0][:-5].strip(","))
#
#
# 		for i in range(len(matrix)):
# 			for j in range(len(matrix)):
# 				number = matrix[j][i]
# 				plt.text(i,j,s = str('%01.2f '%(number if number >=0 else -1*number )))
# 		plt.imshow(matrix, cmap=cm.RdYlGn, vmin=-1., vmax=1.)
# 		plt.title(nome)
# 		norm = mpl.colors.Normalize(vmin=-1., vmax=1.)
# 		cmap = mpl.cm.RdYlGn
# 		color = plt.colorbar(cmap=cmap,norm=norm,orientation='vertical')
# 		color.set_ticks([-1,1],update_ticks=True)
# 		color.set_ticklabels(["errado","certo"])
#
#
# 		if(save):
# 			plt.savefig(path+nome)
# 		if(show):
# 			plt.show()
