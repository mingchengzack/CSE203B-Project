import torch

def MSE_loss(preds,labels):
    return ((preds-labels)**2).mean()


U=torch.load('U.pt')
V_T=torch.load('V_T.pt')
U_gt=torch.load('U_gt.pt')
V_T_gt=torch.load('V_T_gt.pt')

print('MSE U: '+str(float(MSE_loss(U,U_gt))))
print('MSE V_T: '+str(float(MSE_loss(V_T,V_T_gt))))