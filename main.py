import numpy as np
import time
import string
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import linear_model
from sklearn import decomposition
from scipy import stats
import torch
from torch import nn
import torch.optim as optim

cuda=torch.device('cuda')

num_cluster=20

def MSE_loss(preds,labels):
    return torch.sqrt(((preds-labels)**2).sum())

def Map_Labels(V_T, clusters,labels, method):
    pred = np.zeros(len(labels))
    map_label = dict()

    # Find the mode true label in each cluster and make it the prediction
    if method == 'Mode':
        for c in range(len(V_T)):
            count = np.array([labels[i] for i in np.nditer(np.where(clusters == c))])
            mode = stats.mode(count)
            map_label[c] = mode[0][0]
    # Use Kuhn-Munkres algorithm to assign (cost = true label that are not equal to assigned label)
    elif method == 'KM':
        cost = [[] for i in range(num_cluster)]
        m = Munkres()
        for i in range(num_cluster):
            for j in range(num_cluster):
                c = 0
                for k in range(len(labels)):
                    if clusters[k] == i and labels[k] != j:
                        c += 1
                cost[i].append(c)

        assign = m.compute(cost)
        for cluster, label in assign:
            map_label[cluster] = label

    # assign mapping
    for i in range(len(labels)):
        pred[i] = map_label[clusters[i]]
    return pred

def Predict_and_Eval(V_T,labels):
    # V_T shape: num_cluster * num_document, labels shape: 1* num_document
    clusters = np.argmax(V_T, axis=0)
    pred = Map_Labels(V_T, clusters, labels, 'Mode')
    acc = sum(pred == labels) / len(labels)
    return acc

def NMF_sklearn(matrix,labels):
    matrix=matrix.cpu().to_dense().numpy()
    startTime = time.time()
    model=decomposition.NMF(solver='mu',n_components=num_cluster,init='random',verbose=True)
    U=model.fit_transform(matrix)
    V_T=model.components_
    acc=Predict_and_Eval(V_T,labels)
    U=torch.FloatTensor(U)
    V_T=torch.FloatTensor(V_T.copy())
    pred=torch.mm(U,V_T)
    loss_train=MSE_loss(pred,torch.FloatTensor(matrix))
    print("Time elapsed %f" % (time.time() - startTime))
    print("train loss is: %f, evaluation accuracy is: %f" %(float(loss_train),float(acc)))
    torch.save(torch.FloatTensor(U),'U_gt.pt')
    torch.save(torch.FloatTensor(V_T),'V_T_gt.pt')

def SVD_sklearn(matrix,labels):
    matrix=matrix.cpu().to_dense().numpy()
    model=decomposition.TruncatedSVD(n_components=num_cluster,algorithm='arpack')
    U=model.fit_transform(matrix)
    V_T=model.components_
    acc=Predict_and_Eval(V_T,labels)
    U=torch.FloatTensor(U)
    V_T=torch.FloatTensor(V_T.copy())
    pred=torch.mm(U,V_T)
    loss_train=MSE_loss(pred,torch.FloatTensor(matrix))
    print("train loss is: %f, evaluation accuracy is: %f" %(float(loss_train),float(acc)))

def CNMF_Pymf(matrix,labels):
    matrix=matrix.cpu().to_dense().numpy()
    model = CNMF(matrix, num_bases=num_cluster)
    startTime = time.time()
    model.H = np.random.rand(num_cluster, len(labels))
    model.G = np.random.rand(len(labels), num_cluster)
    # model.W = np.dot(matrix[:, :], model.G)
    model.factorize(niter=5,compute_err=False)
    U=model.W
    V_T=model.H
    acc = Predict_and_Eval(V_T, labels)
    U = torch.FloatTensor(U)
    V_T = torch.FloatTensor(V_T.copy())
    pred = torch.mm(U, V_T)
    loss_train = MSE_loss(pred, torch.FloatTensor(matrix))
    print("Time elapsed %f" % (time.time() - startTime))
    print("train loss is: %f, evaluation accuracy is: %f" % (float(loss_train), float(acc)))

def CHNMF_Pymf(matrix,labels):
    matrix=matrix.cpu().to_dense().numpy()
    model = CHNMF(matrix, num_bases=num_cluster)
    model.factorize(niter=10)
    U=model.W
    V_T=model.H
    acc = Predict_and_Eval(V_T, labels)
    U = torch.FloatTensor(U)
    V_T = torch.FloatTensor(V_T.copy())
    pred = torch.mm(U, V_T)
    loss_train = MSE_loss(pred, torch.FloatTensor(matrix))
    print("train loss is: %f, evaluation accuracy is: %f" % (float(loss_train), float(acc)))

#Define latent facor model in PyTorch style
class NMF_SGD(nn.Module):
    def __init__(self,num_document,num_feature,num_cluster):
        super(NMF_SGD,self).__init__()
        self.U=nn.Parameter(0.001*torch.rand(num_feature,num_cluster))
        self.V_T=nn.Parameter(0.001*torch.rand(num_cluster,num_document))
    
    def forward(self):
        # self.U=nn.Parameter(self.U.clamp(min=0))
        # self.V_T=nn.Parameter(self.V_T.clamp(min=0))
        return torch.mm(self.U,self.V_T)

class CNMF_SGD(nn.Module):
    def __init__(self,num_document,num_feature,num_cluster,matrix):
        super(CNMF_SGD,self).__init__()
        self.X=matrix
        self.U=nn.Parameter(0.01*torch.rand(num_document,num_cluster))
        self.V_T=nn.Parameter(0.01*torch.rand(num_cluster,num_document))
    
    def forward(self):
        # self.U=nn.Parameter(self.U.clamp(min=0))
        # self.V_T=nn.Parameter(self.V_T.clamp(min=0))
        return torch.mm(self.X,torch.mm(self.U,self.V_T))

#Training functions to conduct gradient descent in Latent Factor Model and DenseNET
def NMF_training(matrix,labels,method):
    num_document=matrix.shape[1]
    num_feature=matrix.shape[0]

    #Specify the model, learning rate and optimizer
    if method=='NMF_SGD':
        model=NMF_SGD(num_document,num_feature,num_cluster).to(cuda)
        learning_rate=5e-4
        # optimizer=optim.SGD(model.parameters(),lr=learning_rate)
        optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    elif method=='CNMF_SGD':
        model=CNMF_SGD(num_document,num_feature,num_cluster,matrix).to(cuda)
        learning_rate=1e-3
        # optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.8,weight_decay=1e-5)
        optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    
    #Define loss function
    loss_func=MSE_loss

    train_epochs=2000
    for epoch in range(train_epochs):
        startTime=time.time()

        optimizer.zero_grad() #Zero gradient to avoid accumulating
        preds=model() #Forward
        loss_train=loss_func(preds,matrix) #Compute loss on training set
        loss_train.backward() #Back propagation
        optimizer.step() #Update parameters

        with torch.no_grad():
            for name, param in model.named_parameters():
                param.clamp_(min=0)

        # Get V_T and convert to numpy array
        for name, param in model.named_parameters():
            if name=='V_T':
                V_T=param.data.cpu().detach().numpy()
        
        for i in range(V_T.shape[0]):
            for j in range(V_T.shape[1]):
                if(V_T[i,j]<0):
                    print('here')

        if epoch<5 or epoch%10==0:
            #Predict label and compute accuracy
            acc=Predict_and_Eval(V_T,labels)
            print("Epoch: %d, train loss is: %f, evaluation accuracy is: %f. Time elapsed %f" %(epoch,float(loss_train),float(acc),time.time()-startTime))
    
    for name, param in model.named_parameters():
        if name=='V_T':
            torch.save(param.data,'V_T.pt')
        if name=='U':
            torch.save(param.data,'U.pt')

def NMF(matrix,labels,method):
    
    #Perform NMF with different methods
    if method=='NMF_SGD':
        MSE=NMF_training(matrix,labels,'NMF_SGD')
    elif method=='CNMF_SGD':
        MSE=NMF_training(matrix,labels,'CNMF_SGD')
    elif method=='NMF_sklearn':
        NMF_sklearn(matrix,labels)
    elif method=='NMF_surprise':
        NMF_surprise(matrix,labels)
    elif method=='SVD_sklearn':
        SVD_sklearn(matrix,labels)
    elif method=='CNMF_Pymf':
        CNMF_Pymf(matrix,labels)
    elif method=='CHNMF_Pymf':
        CHNMF_Pymf(matrix,labels)

    # print("MSE of "+method+" on validation set is: %f" %MSE)

if __name__ == "__main__":
    #Read dataset
    # matrix shape: num_document * num_feature, labels: num_document * 1 
    matrix=torch.load('./data/x_train.pt')
    labels=torch.load('./data/y_train.pt')
    labels=labels.numpy()

    matrix=matrix.to(cuda)

    from collections import Counter
    num_cluster_gt=len(Counter(labels))
    num_cluster=num_cluster_gt
    print('Read dataset complete')
    print('Matrix size: '+str(matrix.shape)+', category number: '+str(num_cluster_gt))

# #-------------------------plot statistics of dataset-----------------------------
#     plot_dataset_statistics(dataset)

# #-------------------------NMF SGD-----------------------------
#     NMF(matrix,labels,'NMF_SGD')

#-------------------------CNMF SGD-----------------------------
    NMF(matrix,labels,'CNMF_SGD')

# #-------------------------NMF Sklearn-----------------------------
#     NMF(matrix,labels,'NMF_sklearn')

# #-------------------------SVD Sklearn-----------------------------
#     NMF(matrix,labels,'SVD_sklearn')

# #-------------------------CNMF Pymf-----------------------------
#     NMF(matrix,labels,'CNMF_Pymf')
