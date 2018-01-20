import numpy as np

from random  import shuffle

def svm_loss_naive(w,x,y,reg):
    #Inputs:
    #w:A numpy arrary of shape (D,C) containing weights
    #X:A numpy arrary of shape (N,D) containing a minibatch of data
    #y:a numpy arrary of shape (N,) containing training labels: y[i]=c means
    # that X[i] has label c,where 0<=c < C
    #reg:(float) regularization strength

    #Returns:
    #loss as single float
    #gradient with respect to weights W:an arrary of same shape as W

    dw=np.zeros(w.shape)

    num_classes=w.shape[1]
    num_train=x.shape[0]
    loss=0.0
    for i in range(num_train):
        scores=x[i].dot(w)
        correct_class_score=scores[y[i]]
        for j in range(num_classes):
            if j==y[i]:
                continue
            margin=scores[j]-correct_class_score+1
            
            if margin>0:
                loss+=margin
                dw[:,y[i]]-=x[i,:]
                dw[:,j]+=x[i,:]
    loss/=num_train
    dw/=num_train

    loss+=0.5*reg*np.sum(w*w)
    dw+=reg*w

    return loss,dw

def svm_loss_vectorized(w,x,y,reg):
    loss=0.0
    num_train=x.shape[0]
    dw=np.zeros(w.shape)
    scores=np.dot(x,w)
    correct_class_scores=scores[np.arange(num_train),y]
    correct_class_scores=np.reshape(correct_class_scores,(num_train,-1))
    margin=scores-correct_class_scores+1.0
    margin[np.arange(num_train),y]=0.0
    margin[margin<=0]=0.0
    loss+=np.sum(margin)/num_train
    loss+=0.5*reg*np.sum(w*w)

    margin[margin>0]=1.0
    row_sum=np.sum(margin,axis=1)
    margin[np.arange(num_train),y]=-row_sum
    dw=1.0/num_train*np.dot(x.t,margin)+reg*w

    return loss,dw


#if __name__ == '__main__':

