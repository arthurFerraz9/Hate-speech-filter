import arff, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


#[TODO] Loading Dataset - should separate Train, Validate and Train, for Now only using Train

file = arff.load(open('./database/OffComBR2.arff', 'r'))

dataset = np.array(file['data'])

X = [data[1] for data in dataset]
Y = [data[0] for data in dataset]


# Define a Pipeline that:
# vect -> turn text into Bag of Words
# tfidf -> convert ocurrences to frequency
# clf -> choose any classifier
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
 ])

text_clf.fit(X, Y)
predicted = text_clf.predict(X)

print(np.mean(predicted == Y))



