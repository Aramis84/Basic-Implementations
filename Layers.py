import numpy as np

"""
Implementation of the forward and backward passes of various transformations in a fully connected network
Affine transformation Wx + b
Activation ReLu
Batch Normalization
Dropout
"""


def affine_forward(X, w, b):
    """
    computes forward pass of a hidden layer
    Inputs :X input data -> Nx(d1,d2,...,dk) matrix  
            w weights -> DxK matrix where K is the number of neurons in the layer
            b bias terms -> 1D array of K elements
            
    Outputs : out output of the layer -> NxK matrix
              cache tuple -> (X, w, b)  
    """
    N = X.shape[0]
    if len(X.shape) > 2:
        Xnew = X.reshape((N,np.prod(X.shape[1:]))) # NxD matrix
    else:
        Xnew = X                         
                    
    out = np.dot(Xnew,w) + b # NxK matrix
    cache = (X,w,b)
    
    return out,cache
    
    
def affine_backward(dout, cache):
    """
    computes backward pass of a layer
    Inputs :dout -> NxK matrix of gradient of loss with respect to affine layer output coming in from upstream.
            cache -> collection of intermediate variables collected during forward pass that are need for back propagation
            
    Outputs : dX -> gradient with respect to X, NxD matrix
              dw -> gradient with respect to w, DxK matrix
              db -> gradient with respect to b, 1D array of K elements
    """
    X, w, b = cache
    N = dout.shape[0]
    K = w.shape[1]
    
    N = X.shape[0]
    if len(X.shape) > 2:
        Xnew = X.reshape((N,np.prod(X.shape[1:]))) # NxD matrix
    else:
        Xnew = X
                         
    dX = np.dot(dout,w.T) # NxD matrix
    dX = dX.reshape(X.shape)
    dw = np.dot(Xnew.T, dout) # DxK matrix 
    db = np.dot(np.ones((1,N)), dout).reshape(K,) # 1D array of K elements
    
    return dX, dw, db


def relu_forward(X):
    """
    computes ReLu activation of a layer
    Inputs :X input data -> any shape
           
    Outputs : out -> same shape as X
              cache tuple -> X  
    """
    out  = np.maximum(0,X)    
    cache = X
    
    return out, cache
    
    
def relu_backward(dout, cache):
    """
    computes backward pass of a layer
    Inputs :dout -> any shape gradient of loss with respect to relu layer output coming in from upstream.
            cache -> collection of intermediate variables collected during forward pass that are need for back propagation
            
    Outputs : dX -> gradient with respect to X (relu layer input), same shape of x stored in cache
    """
    X = cache
    dout[X<0] = 0.0
    dX = dout
    
    return dX
    

def batchnorm_forward(X, gamma, beta, bn_param):
    """
    computes forward pass of a batch normalization layer
    Inputs :X input data -> NxD matrix
            gamma scale parameter -> 1 dimensional of D elements
            beta shift parameter -> 1 dimensional of D elements
            bn_param dictionary of parameters
                mode : 'Train' or 'Test' batch normalization is not used during test
                momentum : scalar weight for computing exponentially decaying running mean and variance
                eps : small scalar to prevent division by zero
                running_mean : running mean, 1 dimensional with D elements
                running_var : running variance, 1 dimensional with D elements
    
    Outputs : out -> same shape as X
              cache -> tuple of values needed in backward pass  
    """
    
    D = X.shape[1]
    eps = bn_param.get('eps', 1e-9) # get the user passed value or get the default if user didn't provide a value
    m = bn_param.get('momentum', 0.9)
    running_mean = bn_param.get('running_mean', np.zeros(D,dtype = X.dtype))
    running_var = bn_param.get('running_var', np.zeros(D,dtype = X.dtype))
    mode = bn_param['mode'].lower()
    if mode != 'train' and mode != 'test':
        raise ValueError('Must specifiy batch normalization mode of operation')
        
   
    out, cache = None, None # for the first batch normalization layer
    if mode == 'train':
                               
        mu = np.mean(X, axis = 0) # curent batch mean
        var = np.var(X, axis = 0) # current batch variance                       
                               
        running_mean = m*running_mean + (1-m)*mu
        running_var = m*running_var + (1-m)*var                       
       
        xhat = (X - mu)/np.sqrt(var+eps)  # NxD matrix after normalization
        out = gamma*xhat + beta # NxD matrix after scaling and shifting 
        cache = (X, mu, var, xhat, gamma, eps)
        
    elif mode == 'test':
        xhat = (X - running_mean)/np.sqrt(running_var+eps) # for test data, use running statistics for normalization
        out = gamma*xhat + beta
        
   
    # update batch norm parameter dictionary
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    
    return out, cache


def batchnorm_backward(dout, cache):
    """
    computes backward pass of a batch normalization layer
    Inputs :dout -> NxD matrix of gradient of loss with respect to batchnorm layer output coming in from upstream.
            cache -> collection of intermediate variables collected during forward pass that are need for back propagation
            
    Outputs : dX -> gradient with respect to X, NxD matrix
              dgamma -> gradient with respect to gamma, 1 dimensional of D elements
              dbeta -> gradient with respect to beta, 1 dimensional of D elements
    """
    
    X, mu, var, xhat, gamma, eps = cache
    N = X.shape[0]
    
    invsqrtvar = 1/np.sqrt(var + eps)
    
    dxhat = dout*gamma # NxD matrix
       
    dvar = np.sum(dxhat*(X-mu), axis = 0)*(-0.5)*((invsqrtvar)**3) # 1 dimensional of D elements
    dmu =  np.sum(dxhat*(-invsqrtvar), axis = 0) + dvar*(-2/N)*np.sum(X-mu, axis = 0) # 1 dimensional of D elements
    
    dX = dxhat*invsqrtvar + dvar*(2/N)*(X-mu) + dmu*(1/N) # NxD matrix
    dgamma = np.sum(dout*xhat, axis= 0)  # 1 dimensional of D elements
    dbeta = np.sum(dout, axis = 0)  # 1 dimensional of D elements
    
    return dX, dgamma, dbeta
    

def dropout_forward(X, dropout_param):
    """
    computes forward pass of a dropout layer
    Inputs :X input data -> any shape
            dropout_param dictionary of parameters
                mode : 'Train' or 'Test' dropout is not used during test
                p : probability of dropout
                    
    Outputs : out -> same shape as X
              cache -> tuple of values needed in backward pass  
    """
    
    p = dropout_param['p']
    mode = dropout_param['mode'].lower()
    if mode != 'train' and mode != 'test':
        raise ValueError('Must specifiy dropout mode of operation')
    
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])
    
    out, mapper = None, None
    if mode == 'train':
        mapper = (np.random.rand(X.shape[0], X.shape[1])<p)/p # divided by p so that nothing changes during testing phase
        out = X*mapper # nodes which have probability less than p are dropped
        
    elif mode == 'test':
        out = X # since we implemented inverted dropout, nothing in the input changes in the test phase
            
    
    cache = (dropout_param, mapper)
    return out, cache
    

def dropout_backward(dout, cache):
    """
    computes backward pass of a dropout layer
    Inputs :dout -> any shape of gradient of loss with respect to dropout layer output coming in from upstream.
            cache -> collection of intermediate variables collected during forward pass that are need for back propagation
            
    Outputs : dX -> gradient with respect to X, input of dropout layer
    """
    
    dropout_param, mapper = cache
    mode = dropout_param['mode']
    
    if mode == 'train':
        dX = dout*mapper # same shape as X, the input to the dropout layer
    elif mode == 'test':
        dX = dout # no dropout in test phase, so incoming gradient is passed along
        
    return dX        


def svm_loss(S,y):
    """
    Computes the multiclass svm loss 
    Inputs :S score matrix -> NxK matrix where each row has the scores for each class for a particular training instance
            y labels -> 1 dimensional array containing N elements each corresponding to the correct class index
            
    Outputs :loss -> the value of the loss function (scalar)
             grad -> gradient of the loss with respect to the input scores NxK matrix    
    """
    N = S.shape[0]
    correct_class_scores = S[range(N),y].reshape(-1,1)
    scores_intermediate = S - correct_class_scores + 1  # NxK matrix
    scores_intermediate[range(N),y] = 0
    loss = np.sum(np.sum(np.maximum(0, scores_intermediate), axis =1))/N # scalar
    
    temp = np.zeros(S.shape) # NxK matrix
    temp[scores_intermediate>0] = 1
    temp[range(N), y] = -np.sum(temp, axis = 1)
    grad = temp/N  # NxK matrix
    
    return loss, grad


def softmax_loss(S,y):
    """
    Computes the softmax cross-entropy loss 
    Inputs :S score matrix -> NxK matrix where each row has the scores for each class for a particular training instance
            y labels -> 1 dimensional array containing N elements each corresponding to the correct class index
            
    Outputs :loss -> the value of the loss function (scalar)
             grad -> gradient of the loss with respect to the input scores NxK matrix    
    """
    N  = S.shape[0]
    shifted_scores = S - np.max(S, axis=1, keepdims = True)
    prob = np.exp(shifted_scores)/np.sum(np.exp(shifted_scores), axis = 1, keepdims = True) # NxK matrix
    true_class_neg_logptob = -np.log(prob[range(N),y])
    loss = np.sum(true_class_neg_logptob)/N
    
    temp = prob # NxK matrix
    temp[range(N),y] -= 1
    
    grad = temp/N
    
    return loss, grad    
    


# In[ ]:



