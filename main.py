from sklearn.naive_bayes import MultinomialNB
import data_utils
import pipeline as pip
from sklearn.metrics import accuracy_score

data, labels = data_utils.load_data_set_from_arff('./database/OffComBR2.arff')
yes_ratio = data_utils.get_yes_ratio(labels)
print("Distribuição de classes:\n\t'sim': {:.3f}\n\t'não': {:.3f}".format(yes_ratio, 1 - yes_ratio))

# Separate into train and test datasets maintaining proportion
X_train, X_test, y_train, y_test = data_utils.split_test_train(data, labels, test_prop=0.2)

# [TODO] Declarar outros classificadores
classifiers = {"Naive-Bayes" : MultinomialNB()}

# [TODO] Pensar em possíveis processamentos de dados

for classifier_name, classifier in classifiers.items():
    print(classifier_name)

    # Pipeline convert to a frequency-based bag of words and append classifier
    pipeline = pip.get_pipeline(classifier)


    # [TODO] Tunar parâmetros utilizando dataset de test e validação - ex: crossfold
    # Importante: Não utilizar conjunto de teste para tunar parâmetros - somente utilizar teste para comparar modelos
    pipeline.fit(X_train, y_train)
    predicted = pipeline.predict(X_test)

    # [TODO] Colocar mais medidas de qualidade, em especial as que lidam com datasets desbalanceados
    test_accuracy = accuracy_score(y_test, predicted)
    print(test_accuracy)