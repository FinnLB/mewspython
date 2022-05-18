import os.path
import pickle
import pandas as pd
import scipy.stats
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.pipeline import FeatureUnion
from sklearn import svm


corpus = pd.read_csv("data_ordinal.csv")


def test(predictions, gold, pearson=False):
    if pearson:
        print("pearson correlation: ", scipy.stats.pearsonr(predictions, gold))
        predictions = [round(val) for val in predictions]

    print("accuracy: ", accuracy_score(predictions, gold))
    print("qwk: ", cohen_kappa_score(predictions, gold, weights="quadratic"), "\n")
    confusion_matrix = pd.crosstab(predictions, gold, rownames=['predictions'], colnames=['rater 1'])
    print(confusion_matrix)


def train_test(column: str, classifier=svm.SVC(), regressor=svm.SVR()):
    print(column)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(corpus['doc'], corpus[column], test_size=0.2,
                                                                        random_state=3)

    # create features
    wcv = CountVectorizer(analyzer='word', ngram_range=(1, 3))
    ccv = CountVectorizer(analyzer='char', ngram_range=(2, 5))
    vocab = FeatureUnion([('word_ngram_counts', wcv), ('char_ngram_counts', ccv)])
    vocab.fit(corpus['doc'])
    train_x_mat = vocab.transform(train_x)
    test_y_mat = vocab.transform(test_x)

    print("\n", "Classifier")

    if os.path.exists("trained_classifiers/" + column + ".svc"):
        # load classifier
        classifier = pickle.load(open("trained_classifiers/" + column + ".svc", "rb"))
    else:
        # train classifier
        classifier.fit(train_x_mat, train_y)
        # save classifier
        pickle.dump(classifier, open("trained_classifiers/" + column + ".svc", "wb"))

    # test classifier
    predictions_c = classifier.predict(test_y_mat)
    test(predictions_c, test_y)

    print("\n", "Regressor")

    if os.path.exists("trained_classifiers/" + column + ".svr"):
        # load regressor
        regressor = pickle.load(open("trained_regressors/" + column + ".svr", "rb"))
    else:
        # train regressor
        regressor.fit(train_x_mat, train_y)
        # save regressor
        pickle.dump(regressor, open("trained_regressors/" + column + ".svr", "wb"))

    # test regressor
    predictions_r = regressor.predict(test_y_mat)
    test(predictions_r, test_y, pearson=True)


def inter_annotator_agreement(column1: str, column2: str):
    print("iaa - qwk: ", cohen_kappa_score(corpus[column1], corpus[column2], weights="quadratic"))


if __name__ == '__main__':
    last_c1_column = None
    for column in corpus.columns:
        if column.startswith('Code1'):
            last_c1_column = column
            train_test(column)
        elif column.startswith('Code2'):
            inter_annotator_agreement(last_c1_column, column)
            print("---------------------------------------", "\n\n")
