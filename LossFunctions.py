from __future__ import division
import numpy as np
from random import randrange

def svm_loss(X, W, y, reg):
    """
    Implements the max margin mutliclass loss for SVM. The bias terms are incorporated into the paramter matrix W. Note every 
    instance in the training data set must have a trailing 1 added to take care of the bias during matrix multiplication
    
    Inputs : X training data -> Nx(D+1)
             W parameter matrix -> K x (D+1), where K is the number of classes
             y training data labels -> (N,) array
             reg -> regularization parameter
             
    Outputs : the max margin loss, the gradient of the loss with respect to the parameter matrix W
    """
    # computing the loss
    N = X.shape[0]
    scores  = np.dot(X, W.T) # NxK matrix with each row containing the scores for all classes of that training example
    scores_correct_class = scores[range(N), y].reshape(-1,1)
    loss_intermediate = scores - scores_correct_class + 1  # NxK matrix
    loss_intermediate[range(N), y] = 0
    data_loss = np.sum(np.sum(np.maximum(0,loss_intermediate), axis = 1))/N
    
    reg_loss = 0.5*reg*np.sum(W*W)
    L = data_loss + reg_loss
    
    # computing the gradient with respect to W
    temp = np.zeros(loss_intermediate.shape) # NxK matrix
    temp[loss_intermediate>0] = 1
    temp[range(N), y] = -np.sum(temp, axis = 1) # NxK matrix
       
    dL_W = np.dot(temp.T, X)/N # Kx(D+1)
    dL_W += reg*W # including the gradient from the reularization loss
    
    return L, dL_W   
    


def softmax_loss(X, W, y, reg):
    """
    Implements the cross entropy loss using softmax function. The bias terms are incorporated into the paramter matrix W. Note 
    every instance in the training data set must have a trailing 1 added to take care of the bias during matrix multiplication
    
    Inputs : X training data -> Nx(D+1)
             W parameter matrix -> K x (D+1), where K is the number of classes
             y training data labels -> (N,) array
             reg -> regularization parameter
             
    Outputs : the cross entropy loss, the gradient of the loss with respect to the parameter matrix W
    """
    N = X.shape[0]
    # computing the loss
    scores  = np.dot(X, W.T) # NxK matrix with each row containing the scores for all classes of that training example
    # class probabilites
    prob = np.exp(scores)/np.sum(np.exp(scores),axis = 1, keepdims =  True) # NxK matrix, each row sums to 1
    # negative log probability of true class
    true_class_neglogprob = -np.log(prob[range(N), y])
    # total data loss
    data_loss = np.sum(true_class_neglogprob)/N
    # total regularization loss
    reg_loss = 0.5*reg*np.sum(W*W) # L2 regularization
    # total loss 
    L = data_loss + reg_loss
    
    # computing the gradient with respect to W
    dL_scores = prob
    # at the indices of the correct classes we need to subtract 1 from the probabilities (check derivative of loss with repsect
    # to scores)
    dL_scores[range(N),y] -=1 # NxK matrix
    dL_scores /= N
    
    # gradient of data loss with respect to the parameters using chain rule
    # note the gradient with respect to W must be of the same size as W 
    
    dL_W = np.dot(dL_scores.T, X)  # Kx(D+1)
    dL_W += reg*W # including the gradient from the reularization loss
    
    return L, dL_W   
    

def gradient_check(f, W, analytic_grad, num_dim = 10, h = 1e-5):
    
    """
    function to check analytical and numerical gradients
    
    Inputs: f function whose gradient is to be computed
            W parameter -> the gradient of f is computed with respect to W
            analytic_grad -> gradient computed analytically for comparison
            num_dim -> number of dimensions along which gradient is to be checked
            h -> step to compute numerical derivative
    
    Outputs: returns relative error between numerical and analytical gradients
               
    """
    num_Grad = []
    rel_Error = []
    for ii in range(num_dim):
        
        dims = tuple([randrange(r) for r in W.shape]) # choosing a dimension randomly among all 
        past_val = W[dims]
        W[dims] = past_val + h
        f_val_incr = f(W)
        
        W[dims] = past_val - h
        f_val_decr = f(W)
        W[dims] = past_val
        
        numeric_grad  = (f_val_incr - f_val_decr)/(2*h)
        relative_error = abs(analytic_grad[dims] - numeric_grad)/(abs(analytic_grad[dims]) + abs(numeric_grad))
        rel_Error.append(relative_error)
        num_Grad.append(numeric_grad)
        
        print('Relative error : %.2f') %(relative_error)
    return num_Grad, rel_Error






