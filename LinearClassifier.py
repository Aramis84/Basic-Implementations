import numpy as np
from LossFunctions import *

class LinearClassifier(object):
    """ Implementing a linear classifier. The classifier can use either the SVM max margin loss function or the softmax cross
        entropy loss function. """
   
    def __init__(self):
        self.W = None
        
    def train(self, X, y, learning_rate = 1e-3, reg = 1e-5, num_iter = 200, batch_size = 200, verbose = False):
        """
        Trains a linear classifer using mini-batch gradient descent
        
        Inputs :X (training data) -> NxD matrix
                y (training labels) -> (N,) array containing class labels
                learning_rate -> float, step size for gradient descent
                reg -> float, magnitude of regularization
                num_iter -> int, number of optimization steps
                batch_size -> int, training set size for wach step
                verbose -> boolean, for displaying training progress and loss
                
        Outputs : Loss function at each iteration
        
        """
        
        N = X.shape[0] # number of examples
        D = X.shape[1] # number of features #note that this function assumes bias term is included in W , so D is number of 
        # original features plus 1
        K = np.max(y) + 1 # number of classes, y contains index of the correct class beginning with 0, hence need to add 1
        
        if self.W is None:
            self.W = 0.001*np.random.rand(K,D) # for each class there is a row in the parameter matrix
        
        L_accu = []
        for ii in range(num_iter):
            # getting random mini batches
            ind = np.random.choice(N,batch_size)
            X_batch = X[ind,:]
            y_batch = y[ind]
            
            # getting loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            L_accu.append(loss)
            
            if verbose and ii %50 ==0:
                print 'Loss in iteration %d is %.4f' %(ii, loss)
                
                
            self.W -= learning_rate*grad
            
        return L_accu
    
    
    def loss(self, X_batch, y_batch, reg):
        pass
             
    def predict(self, X):
        
        scores = np.dot(X, (self.W).T)
        y_pred = np.argmax(scores,axis = 1)
        
        return y_pred
        
class LinearSVM(LinearClassifier): # inherits train, predict methods but replaces the loss method
    """
    This class inherits the LinearClassifier class and uses max margin multiclass SVM loss function"""
    def loss(self, X_batch, y_batch, reg):
        return svm_loss(X_batch, self.W, y_batch, reg)
    

class Softmax(LinearClassifier): # inherits train, predict methods but replaces the loss method
    """
    This class inherits the LinearClassifier and uses cross entropy loss function"""
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss(X_batch, self.W, y_batch, reg)  
        

