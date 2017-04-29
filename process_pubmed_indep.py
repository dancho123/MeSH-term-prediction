import os, sys
import re, time
import numpy as np
from collections import Counter, defaultdict
import cPickle

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt

'''
Here we read in data extracted from the pubmed database and train machine
learning models to predict the MeSH headings.
'''

def main():

    path = '/Users/xflorian/Downloads/tmp/'

    print('Read data'),
    time0 = time.time()
    y, list_of_authors, list_of_titles, list_of_abstracts = read_data(path)
    print 'done after %d seconds' % (time.time() - time0)

    print('Create correlation matrix'),
    time0 = time.time()
    create_matrix(list_of_abstracts)
    print 'done after %d seconds' % (time.time() - time0)
    sys.exit()

    print('prepare features'),
    # produce a (samples, feature) matrix
    time0 = time.time()
    X = prepare_data(list_of_authors, list_of_titles, list_of_abstracts)
    print 'done after %d seconds' % (time.time() - time0)

    print('transform class labels to numerals'),
    time0 = time.time()
    mlb = MultiLabelBinarizer()
    y_numeric = mlb.fit_transform(y) 
    print 'done after %d seconds' % (time.time() - time0)

    print('Split data'),
    time0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.25, random_state=42)
    print 'done after %d seconds' % (time.time() - time0)

    '''
    The OneVsRestClassifier strategy involves training a single classifier per class, with the samples of that class 
    as positive samples and all other samples as negatives. 
    Note that the OneVsRestClassifier does not take into account dependencies between MeSH terms, meaning if an input 
    had two MeSH, it is equal to having two identical inputs with one MeSH term each. Relations between MeSH terms should 
    help the classification, for example papers with MeSH term 'cancer' might have a higher probability to also have a 
    MeSH term 'cancer treatment'. This extra information is currently not used.
    '''

    # Test different classifiers 
    print "test Logistic Regression"
    from sklearn.linear_model import LogisticRegression
    clf = OneVsRestClassifier(LogisticRegression())
    test_clf(clf, X_train, y_train, X_test, y_test, mlb)
    print "done"

    print "test MultinomialNB"
    from sklearn.naive_bayes import MultinomialNB
    clf = OneVsRestClassifier(MultinomialNB())
    test_clf(clf, X_train, y_train, X_test, y_test, mlb)
    print "done"

    print "test LinearSVC"
    from sklearn import svm
    clf = OneVsRestClassifier(svm.LinearSVC(random_state=0))
    test_clf(clf, X_train, y_train, X_test, y_test, mlb)
    print "done"

    print "test DecisionTreeRegressor"
    from sklearn.tree import DecisionTreeRegressor
    clf = OneVsRestClassifier(DecisionTreeRegressor(random_state=0))
    test_clf(clf, X_train, y_train, X_test, y_test, mlb)
    print "done"

    return 


def test_clf(clf, X_train, y_train, X_test, y_test, mlb):
    print('Fit data'),
    time0 = time.time()
    clf.fit(X_train, y_train)
    print 'done after %d seconds' % (time.time() - time0)

    print('Predict data'),
    time0 = time.time()
    y_pred = clf.predict(X_test)
    print 'done after %d seconds' % (time.time() - time0)
    #for i, pred in enumerate(y_pred):
    #    print pred
    #    print y_test[i]
    #    print "\n"

    print('Evaluate data'),
    time0 = time.time()
    evaluate = [1 if (prediction == y_test[i]).all() else 0 for i, prediction in enumerate(y_pred)]
    report = classification_report(y_test, y_pred, target_names=mlb.classes_)
    print 'done after %d seconds' % (time.time() - time0)
    print "prediction for %d papers" % len(y_pred)
    print "fraction of correct prediction for all MeSH terms (including multiple MeSH terms) = ", float(sum(evaluate))/len(y_pred)
    print report
    return


def read_data(path):
    list_of_authors = cPickle.load(open('%sauthors.pickle' % path, 'rb'))
    list_of_titles = cPickle.load(open('%stitles.pickle' % path, 'rb'))
    list_of_abstracts = cPickle.load(open('%sabstracts.pickle' % path, 'rb'))
    y = cPickle.load(open('%sy.pickle' % path, 'rb'))
    return y, list_of_authors, list_of_titles, list_of_abstracts


def create_matrix(list_of_abstracts):
    tokenizer = RegexpTokenizer(r'\w+')
    list_of_filtered_lists = []
    print "len(list_of_abstracts) = ", len(list_of_abstracts)
    for abstract in list_of_abstracts:
        list_of_words = tokenizer.tokenize(abstract.lower())
        filtered_list_of_words = [word for word in list_of_words if conditions(word)] # remove digits as well
        print len(list_of_filtered_lists)
        list_of_filtered_lists.append(filtered_list_of_words) # assuming we are not interested to distinguish between lower and upper case
    print "number of files = ", len(list_of_filtered_lists)

    result_matrix = np.zeros([len(list_of_filtered_lists), len(list_of_filtered_lists)])
    for i, list1 in enumerate(list_of_filtered_lists):
        # we only need to calculate half of this symmetric matrix
        for j in range(0, i):
            result_matrix[i][j] = len(list(set(list1).intersection(list_of_filtered_lists[j])))
            print i, j, result_matrix[i][j]
    print "result_matrix = ", result_matrix
    cPickle.dump(result_matrix, open('/Users/xflorian/Downloads/tmp/matrix.pickle', 'wb')) 
    return result_matrix


# Filters for the words which enter the analysis
def conditions(word):
    return (word not in stopwords.words('english') and not word.isdigit() and len(word) > 1)


def prepare_data(list_of_authors, list_of_titles, list_of_abstracts):
    # We calculate the tf-idf for each feature
    vectorizer = TfidfVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = "english", max_features = 5000) 
    train_data_features1 = vectorizer.fit_transform(list_of_abstracts)
    train_data_features2 = vectorizer.fit_transform(list_of_titles)
    train_data_features3 = vectorizer.transform(list_of_authors)

    # Numpy arrays are easy to work with, so convert the results to arrays
    train_data_features1 = train_data_features1.toarray()
    train_data_features2 = train_data_features2.toarray()
    train_data_features3 = train_data_features3.toarray()

    X = np.c_[train_data_features1, train_data_features2, train_data_features3]

    return X


if __name__ == "__main__":
    main()
