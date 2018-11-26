from sklearn.pipeline import Pipeline
from get_stop_words import get_stop_words
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
def get_pipeline(classifier):
    return Pipeline([
     ('vect', CountVectorizer(stop_words= get_stop_words('stop_words.txt'))),
     ('tfidf', TfidfTransformer()),
     ('clf', classifier),
 ])
