import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    newsgroups_train = fetch_20newsgroups(data_home="./data/", subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(data_home="./data/", subset='test', remove=('headers', 'footers', 'quotes'))
    vectorizer = TfidfVectorizer(max_features=5000)
    # x_train, x_test are scipy sparse csr matrix
    x_train = vectorizer.fit_transform(newsgroups_train.data) 
    x_test = vectorizer.fit_transform(newsgroups_test.data)
    y_train = newsgroups_train.target
    y_test = newsgroups_test.target

    # convert x_train, x_test to torch sparse tensor
    x_train_coo=x_train.tocoo()
    x_train_tensor=torch.sparse.FloatTensor(torch.LongTensor([x_train_coo.row.tolist(),x_train_coo.col.tolist()]),torch.FloatTensor(x_train_coo.data.astype(np.float32)))
    x_test_coo=x_test.tocoo()
    x_test_tensor=torch.sparse.FloatTensor(torch.LongTensor([x_test_coo.row.tolist(),x_test_coo.col.tolist()]),torch.FloatTensor(x_test_coo.data.astype(np.float32)))
    # convert y_train,y_test to normal tensor
    y_train_tensor=torch.LongTensor(y_train)
    y_test_tensor=torch.LongTensor(y_test)
    
    # transpose
    x_train_tensor=torch.transpose(x_train_tensor,0,1)
    x_test_tensor=torch.transpose(x_test_tensor,0,1)


    torch.save(x_train_tensor,'./data/x_train.pt')
    torch.save(x_test_tensor,'./data/x_test.pt')
    torch.save(y_train_tensor,'./data/y_train.pt')
    torch.save(y_test_tensor,'./data/y_test.pt')
if __name__ == "__main__":
    main()
