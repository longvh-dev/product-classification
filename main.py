import argparse

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import pickle
import os
import pandas as pd
import numpy as np

from utils import text_preprocess


def get_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_path', type=str, default='../data/data.txt')
    parser.add_argument('--test_percent', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)

    # save
    parser.add_argument('--model_path', type=str, default='../save')

    # others
    parser.add_argument('--is_preprocess', type=bool, default=True)
    parser.add_argument('--is_train', type=bool, default=True)
    parser.add_argument('--is_evaluate', type=bool, default=True)
    args = parser.parse_args()
    return args

def load_data(path):
    print('Loading data from', path)
    f = open(path, "r")
    products, categories = [], []
    while True:
        line = f.readline()
        if not line:
            break

        category = line.split(" ", 1)[0]
        product = ''.join(line.split(" ", 1)[1:]).replace("\n", "")

        categories.append(category)
        products.append(product)
    return [products, categories]


def split_data(df, test_percent, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        df['products'], df['categories'], test_size=test_percent, random_state=random_state)
    return X_train, X_test, y_train, y_test


def encode_target(y_train, y_test):
    # encode target
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    # print(list(label_encoder.classes_)[0:5], 'n')

    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)
    return y_train, y_test


# train naive bayes model
def train_naive_bayes(X_train, y_train, model_path):
    start_time = time.time()
    print('Start training Naive Bayes model...')
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                  max_df=0.8,
                                                  max_features=None)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB())
                         ])
    text_clf = text_clf.fit(X_train, y_train)

    train_time = time.time() - start_time
    print('Done training Naive Bayes in', train_time, 'seconds.')

    # Save model
    pickle.dump(text_clf, open(os.path.join(model_path, "naive_bayes.pkl"), 'wb'))


def evaluate(X_test, y_test, model_path):
    # Naive Bayes
    nb_model = pickle.load(open(os.path.join(model_path, "naive_bayes.pkl"), 'rb'))
    y_pred = nb_model.predict(X_test)
    print('Naive Bayes, Accuracy =', np.mean(y_pred == y_test))

    # SVM
    svm_model = pickle.load(open(os.path.join(model_path, "svm.pkl"), 'rb'))
    y_pred = svm_model.predict(X_test)
    print('SVM, Accuracy =', np.mean(y_pred == y_test))



# ----------------------------------------------------------
# train svm model

# start_time = time.time()
# text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
#                                               max_df=0.8,
#                                               max_features=None)),
#                      ('tfidf', TfidfTransformer()),
#                      ('clf', SVC(gamma='scale'))
#                      ])
# text_clf = text_clf.fit(X_train, y_train)

# train_time = time.time() - start_time
# print('Done training SVM in', train_time, 'seconds.')

# # Save model
# pickle.dump(text_clf, open(os.path.join(MODEL_PATH, "svm.pkl"), 'wb'))


# # Naive Bayes
# nb_model = pickle.load(open(os.path.join(MODEL_PATH, "naive_bayes.pkl"), 'rb'))
# y_pred = nb_model.predict(X_test)
# print('Naive Bayes, Accuracy =', np.mean(y_pred == y_test))
#
# # SVM
# svm_model = pickle.load(open(os.path.join(MODEL_PATH, "svm.pkl"), 'rb'))
# y_pred = svm_model.predict(X_test)
# print('SVM, Accuracy =', np.mean(y_pred == y_test))


def main():
    args = get_args()

    data = load_data(args.data_path)
    df = pd.DataFrame(list(zip(data[0], data[1])),
                      columns = ['products', 'categories'])

    X_train, X_test, y_train, y_test = split_data(df, args.test_percent, args.random_state)

    # X_train = text_preprocess(X_train)
    # if args.is_preprocess:
    #     print('Preprocessing data...')
    #     X_train = [text_preprocess(x) for x in X_train]

    y_train, y_test = encode_target(y_train, y_test)

    if args.is_train:
        train_naive_bayes(X_train, y_train, args.model_path)
    if args.is_evaluate:
        evaluate(X_test, y_test, args.model_path)

if __name__ == "__main__":
    main()
