from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import data_utils
import metrics_utils
import pipeline as pip
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np

data, labels = data_utils.load_data_set_from_arff('./database/OffComBR2.arff')
yes_ratio = data_utils.get_yes_ratio(labels)
print("Distribuição de classes:\n\t'sim': {:.3f}\n\t'não': {:.3f}\n".format(yes_ratio, 1 - yes_ratio))

data, X_test, labels, y_test = data_utils.split_test_train(data, labels, test_prop=0.2)

#Its needed to add clf__ before parameter so the pipeline can interpret it
naive_bayes_params = {"clf__alpha" : np.arange(0.1, 10.1, 0.25),
                      "clf__fit_prior" : [True, False]}

knn_params = {"clf__n_neighbors" : np.arange(1,20,2),
              "clf__metric" : ['cosine', 'euclidean', 'manhattan']}

# [TODO] Declarar outros classificadores e seus parâmetros de variação
classifiers = {"Naive-Bayes" : {"classifier" : MultinomialNB(), "params" : naive_bayes_params},
               "kNN" : {"classifier" : KNeighborsClassifier(), "params" : knn_params}}


for classifier_name, classifier_data in classifiers.items():
    print(classifier_name)

    # Pipeline convert to a frequency-based bag of words and append classifier
    pipeline = pip.get_pipeline(classifier_data['classifier'])

    # GridSearch do a cv-times Cross-Validation varying parameters and get the best one
    clf = GridSearchCV(pipeline, classifier_data['params'], cv=10, n_jobs=-1, iid=True, scoring="balanced_accuracy")

    clf.fit(data, labels)
    print("Melhor conjunto de parâmetros: ", clf.best_estimator_.steps[-1][1])

    predicted = clf.predict(X_test)

    test_accuracy = accuracy_score(y_test, predicted)
    print("Acurácia: {:.3f}".format(test_accuracy), end="\n\n")

    # [TODO] Colocar mais medidas de qualidade, em especial as que lidam com datasets desbalanceados