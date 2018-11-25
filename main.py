from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import data_utils
import metrics_utils
import pipeline as pip
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

data, labels = data_utils.load_data_set_from_arff('./database/OffComBR2.arff')
yes_ratio = data_utils.get_yes_ratio(labels)
print("Distribuição de classes:\n\t'sim': {:.3f}\n\t'não': {:.3f}\n".format(yes_ratio, 1 - yes_ratio))

# [TODO] Declarar outros classificadores
classifiers = {"Naive-Bayes" : MultinomialNB(),
               "k-NN" : KNeighborsClassifier()}

# [TODO] Pensar em possíveis processamentos de dados

for classifier_name, classifier in classifiers.items():
    print(classifier_name)

    # Pipeline convert to a frequency-based bag of words and append classifier
    pipeline = pip.get_pipeline(classifier)

    # [TODO] Tunar parâmetros utilizando dataset de test e validação - ex: crossfold
    qty_cross_validations = 10
    kf = KFold(n_splits = qty_cross_validations, shuffle=True)
    accuracy_collection = []

    print("Usando {:d}-fold-cross-validation...".format(qty_cross_validations))
    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)

        # [TODO] Colocar mais medidas de qualidade, em especial as que lidam com datasets desbalanceados
        accuracy_collection.append(accuracy_score(y_test, predicted))

    accuracy_mean = metrics_utils.calculate_accuracy_mean(accuracy_collection)
    print("Acurácia: {:.3f}".format(accuracy_mean), end="\n\n")
