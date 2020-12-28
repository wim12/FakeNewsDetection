import itertools

import pandas as pd
from sklearn import metrics, pipeline
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
from slugify import slugify

dataset = 'fake_or_real_news.csv'


def loadPanda(dataset):
    loaded_panda = pd.read_csv(dataset)
    print(loaded_panda.shape)
    loaded_panda = loaded_panda.set_index("Unnamed: 0")
    print(loaded_panda.head())
    loaded_panda.title = loaded_panda.title.str.lower()
    loaded_panda.text = loaded_panda.text.str.lower()
    print(loaded_panda.head())
    # remove the URL's present
    loaded_panda.title = loaded_panda.title.str.replace(r'http[\w:/\.]+', '<URL>')
    loaded_panda.text = loaded_panda.text.str.replace(r'http[\w:/\.]+', '<URL>')

    # remove everything except for the characters and the punctuation
    loaded_panda.title = loaded_panda.title.str.replace(r'[^\.\w\s]', '')
    loaded_panda.text = loaded_panda.text.str.replace(r'[^\.\w\s]', '')

    # replacing multiple . with one .
    loaded_panda.title = loaded_panda.title.str.replace(r'[^\.\w\s]', '')
    loaded_panda.text = loaded_panda.text.str.replace(r'[^\.\w\s]', '')

    # adds spaces before and after each .
    loaded_panda.title = loaded_panda.title.str.replace(r'\.', ' . ')
    loaded_panda.text = loaded_panda.text.str.replace(r'\.', ' . ')

    # replaces multiple spaces with single spaces
    loaded_panda.title = loaded_panda.title.str.replace(r'\s\s+', ' ')
    loaded_panda.text = loaded_panda.text.str.replace(r'\s\s+', ' ')

    loaded_panda.title = loaded_panda.title.str.strip()
    loaded_panda.text = loaded_panda.text.str.strip()
    print(loaded_panda.shape)
    print(loaded_panda.head())
    return loaded_panda


def printMets(vectype, y_test, pred, modlename):
    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)
    cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
    plot_confusion_matrix(cm, classes=['FAKE', 'REAL'], model=modlename)
    tfi_f1 = metrics.f1_score(y_test, pred, average='macro')
    print('f1 score: ', tfi_f1)
    tfi_acc = metrics.accuracy_score(y_test, pred)
    print('accuracy: ', tfi_acc)
    tfi_prec = metrics.precision_score(y_test, pred, average='micro')
    print('precision: ', tfi_prec)
    tfi_recall = metrics.recall_score(y_test, pred, pos_label='REAL')
    print('recall: ', tfi_recall)
    classrep = metrics.classification_report(y_test, pred, labels=['FAKE', 'REAL'])
    print(classrep)
    results(vectype, tfi_acc, tfi_f1, tfi_prec, tfi_recall, classrep)


def plot_confusion_matrix(cm, classes, model,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    title = "%s Confusion matrix" % model
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = numpy.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname="%s-conmat.png" % slugify(model))
    plt.show()


fakeRealPanda = loadPanda(dataset)

fakeRealPanda.head()

# Set `y`
y = fakeRealPanda.label

# Drop the `label` column
fakeRealPanda.drop("label", axis=1)


def count_fe(train, test):
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the training data
    count_train = count_vectorizer.fit_transform(X_train)

    # Transform the test set
    count_test = count_vectorizer.transform(X_test)

    print(count_vectorizer.get_feature_names()[:10])


# Make training and test sets
X_train, X_test, y_train, y_test = train_test_split(fakeRealPanda['text'], y, test_size=0.33, random_state=53)

# Initialize the `count_vectorizer`
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data
count_train = count_vectorizer.fit_transform(X_train)

# Transform the test set
count_test = count_vectorizer.transform(X_test)

# Initialize the `tfidf_vectorizer`
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform the training data
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

# Transform the test set
tfidf_test = tfidf_vectorizer.transform(X_test)

# Get the feature names of `tfidf_vectorizer`
print(tfidf_vectorizer.get_feature_names()[-10:])

# Get the feature names of `count_vectorizer`
print(count_vectorizer.get_feature_names()[:10])

count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())

tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())

difference = set(count_df.columns) - set(tfidf_df.columns)
difference

print(count_df.equals(tfidf_df))

count_df.head()
tfidf_df.head()

preproc = [
    StandardScaler(with_mean=False),
    TruncatedSVD()
]

classifiers = [
    MultinomialNB(),
    SGDClassifier(shuffle=True),
    KNeighborsClassifier(),
    RidgeClassifier(),
    DecisionTreeClassifier(),
    LinearSVC(random_state=0),
    RandomForestClassifier(),
    GradientBoostingClassifier()
]


def results(resfile, acc, f1, prec, rec, classrep):
    opfile = open("%s.txt" % resfile, "a+")
    opfile.write(str(classifier))
    opfile.write("f1 score: %f\n" % f1)
    opfile.write('accuracy: %f\n' % acc)
    opfile.write('precision: %f\n' % prec)
    opfile.write('recall: %f\n' % rec)
    opfile.write(classrep)
    opfile.close()


def hyperparams(classifier, parameters, modelname):
    acc_scorer = make_scorer(accuracy_score)
    gridmnb = GridSearchCV(classifier, parameters, scoring=acc_scorer)
    gridmnb = gridmnb.fit(count_train, y_train)
    classifier = gridmnb.best_estimator_
    classifier.fit(count_train, y_train)
    print(classifier, ': count vectorizer')
    pred = classifier.predict(count_test)
    modname = modelname + " Count Vectorizer "
    printMets("Countvec", y_test, pred, modname)

    acc_scorer2 = make_scorer(accuracy_score)
    gridmnb = GridSearchCV(classifier, parameters, scoring=acc_scorer2)
    gridmnb = gridmnb.fit(tfidf_train, y_train)
    classifier = gridmnb.best_estimator_
    classifier.fit(tfidf_train, y_train)
    print(classifier, ': tfidf')
    pred = classifier.predict(tfidf_test)
    modname = modelname + " TFI-DF Vectorizer "
    printMets("Tfidfvec", y_test, pred, modname)
    return classifier


for classifier in classifiers:
    if classifier == classifiers[0]:
        parameters = {
            'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'fit_prior': [True, False]
        }
        modelname = "Multinomial Naive-Bayes"
        classifier = hyperparams(classifier, parameters, modelname)

    elif classifier == classifiers[1]:
        parameters = {
            'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'max_iter': [2000],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
        }
        modelname = "Stochastic Gradient Descent"
        classifier = hyperparams(classifier, parameters, modelname)

    elif classifier == classifiers[2]:
        parameters = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        modelname = "K-Nearest-Neighbours"
        classifier = hyperparams(classifier, parameters, modelname)

    elif classifier == classifiers[3]:
        parameters = {
            'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0],
            'fit_intercept': [True, False],
            'solver': ['sparse_cg', 'sag']
        }
        modelname = "Ridge regression Classifier"
        classifier = hyperparams(classifier, parameters, modelname)

    elif classifier == classifiers[4]:
        parameters = {
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                          27, 28, 29, 30],
            'random_state': [5],
            'criterion': ['gini', 'entropy']
        }
        modelname = "Decision Tree classifier"
        classifier = hyperparams(classifier, parameters, modelname)
    elif classifier == classifiers[5]:
        parameters = {
            'C': [1, 10, 100, 1000]
        }
        modelname = "Linear Support-Vector Classifier"
        classifier = hyperparams(classifier, parameters, modelname)
    elif classifier == classifiers[6]:
        parameters = {
            'bootstrap': [True, False],
            'max_depth': [10, 20, 30, 40, 50],
            'max_features': ['auto', 'sqrt'],
            'min_samples_leaf': [1, 2, 4],
            'min_samples_split': [2, 5, 10],
            'n_estimators': [200, 400, 600, 800, 1000]
        }
        modelname = "Forest of Randomised Decision Trees"
        classifier = hyperparams(classifier, parameters, modelname)
    elif classifier == classifiers[7]:
        parameters = {
            'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
            'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750]
        }
        modelname = "Gradient Boosting Classifier"
        classifier = hyperparams(classifier, parameters, modelname)
    else:
        classifier.fit(count_train, y_train)
        print(classifier, ': count vectorizer')
        pred = classifier.predict(count_test)
        printMets(y_test, pred)
        classifier.fit(tfidf_train, y_train)
        print(classifier, ': tfidf')
        pred = classifier.predict(tfidf_test)
        printMets(y_test, pred)
