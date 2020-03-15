import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    newsgroups_train = fetch_20newsgroups(data_home="./data/", subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(data_home="./data/", subset='test', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train = vectorizer.fit_transform(newsgroups_train.data).toarray()
    y_train = newsgroups_train.target
    x_test = vectorizer.fit_transform(newsgroups_test.data).toarray()
    y_test = newsgroups_test.target

    x_train=np.transpose(x_train)
    y_train=np.transpose(y_train)
    x_test=np.transpose(x_test)
    y_test=np.transpose(y_test)

    np.savez_compressed('data/train.npz', texts=x_train, labels=y_train)
    np.savez_compressed('data/test.npz', texts=x_test, labels=y_test)
if __name__ == "__main__":
    main()
