import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    newsgroups_train = fetch_20newsgroups(data_home="./data/", subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(data_home="./data/", subset='test', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer()
    print(newsgroups_test.data[0])
    x_train = vectorizer.fit_transform(newsgroups_train.data)
    y_train = newsgroups_train.target
    x_test = vectorizer.fit_transform(newsgroups_test.data)
    y_test = newsgroups_test.target

    np.savez_compressed('data/train.npz', texts=x_train, labels=y_train)
    np.savez_compressed('data/test.npz', texts=x_test, labels=y_test)
if __name__ == "__main__":
    main()
