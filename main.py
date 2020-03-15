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
import torch
from torch import nn
import torch.optim as optim

def MSE_loss(preds,labels):
    return ((preds-labels)**2).mean()

def Predict_and_Eval(V_T,labels):
    # V_T shape: num_cluster * num_document, labels shape: 1* num_document
    pass


#Define latent facor model in PyTorch style
class NMF_SGD(nn.Module):
    def __init__(self,num_document,num_feature,num_cluster): #K is the latent vector dimension, alpha_init is used to initialize alpha
        super(NMF_SGD,self).__init__()
        self.U=nn.Parameter(torch.rand(num_feature,num_cluster))
        self.V_T=nn.Parameter(torch.rand(num_cluster,num_document))
    
    def forward(self):
        return torch.mm(self.U,self.V_T)

#Training functions to conduct gradient descent in Latent Factor Model and DenseNET
def NMF_training(matrix,labels,method):
    num_document=matrix.shape[1]
    num_feature=matrix.shape[0]
    num_cluster=20

    #Specify the model, learning rate and optimizer
    model=NMF_SGD(num_document,num_feature,num_cluster)
    learning_rate=1
    optimizer=optim.SGD(model.parameters(),lr=learning_rate,momentum=0.8,weight_decay=5e-4)
    
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

        # Get V_T and convert to numpy array
        for name, param in model.named_parameters():
            if name=='V_T':
                V_T=param.data.detach().numpy()

        #Predict label and compute accuracy
        acc=Predict_and_Eval(V_T,labels)

        if epoch<5 or epoch%5==0:
            print("Epoch: %d, train loss is: %f. Time elapsed %f" %(epoch,float(loss_train),time.time()-startTime))
            # print("Epoch: %d, train loss is: %f, evaluation accuracy is: %f" %(epoch,float(loss_train),float(acc)))

def NMF(matrix,labels,method):
    
    #Perform NMF with different methods
    if method=='SGD':
        MSE=NMF_training(matrix,labels,'SGD')
    
    # print("MSE of "+method+" on validation set is: %f" %MSE)



if __name__ == "__main__":


    #Read dataset
    # matrix shape: num_document * num_feature, labels: num_document * 1 
    matrix=torch.load('./data/x_test.pt')
    labels=torch.load('./data/y_test.pt')
    print('Read dataset complete')
    print('Matrix size: '+str(matrix.shape))

# #-------------------------plot statistics of dataset-----------------------------
#     plot_dataset_statistics(dataset)

#-------------------------NMF SGD-----------------------------
    NMF(matrix,labels,'SGD')

