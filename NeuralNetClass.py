
# coding: utf-8

# In[37]:

from Layers import *
from Layer_Configs import *
from GradientCheck import *
import numpy as np


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss. Input data set has D dimensions (features) and classification is performed over K classes.

    This class does not implement an optimizer. It will interact with a separate solver (provided in the CS 231n class material)
    
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, wt_init_std = 0.001, reg = 0.0):
        
        self.D = input_dim # number of features
        self.H = hidden_dim # number of hidden neurons
        self.K = output_dim # number of classes
        self.std = wt_init_std
        self.W1 = self.std*np.random.randn(self.D,self.H)
        self.b1 = np.zeros((self.H,))
        self.W2 = self.std*np.random.randn(self.H,self.K)
        self.b2 = np.zeros((self.K,))
        self.reg = reg
        
        
        self.params = {}
        self.params['W1'] = self.W1
        self.params['b1'] = self.b1 
        self.params['W2'] = self.W2
        self.params['b2'] = self.b2
        
    
    def loss(self, X, y = None):
        
        N = X.shape[0]
        
        # forward pass
        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        out2, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])
        
        scores = out2
        
        if y is None: # test phase
            return scores
        else:        
            data_loss, dscores = softmax_loss(scores,y)
            reg_loss_W1 = 0.5*self.reg*np.sum(self.params['W1']*self.params['W1'])
            reg_loss_W2 = 0.5*self.reg*np.sum(self.params['W2']*self.params['W2'])
            loss = data_loss + reg_loss_W1 + reg_loss_W2 
            
            #backward pass
            
            dout1, dW2, db2 = affine_backward(dscores, cache2)
            dX, dW1, db1 = affine_relu_backward(dout1,cache1)
            
            reg_grad_W1 = self.reg*self.params['W1']
            reg_grad_W2 = self.reg*self.params['W2']
            
            grads = {}
            grads['W1'] = dW1 + reg_grad_W1
            grads['b1'] = db1
            grads['W2'] = dW2 + reg_grad_W2
            grads['b2'] = db2
            
            return loss, grads      
                

class MultiLayerNet(object):
    
    """
    A multi-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss. Input data set has D dimensions (features) and classification is performed over K classes.

    Both batch normalization and dropout can be used.

    This class does not implement an optimizer. It will interact with a separate solver (provided in the CS 231n class material)
    
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    
    def __init__ (self, input_dim, hidden_dims, output_dim, wt_init_std = 0.001, reg = 0.0, dropout = 0, use_batchnorm = False,                  seed = None, dtype = np.float32):
        
        
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout>0
        self.reg = reg
        self.L = len(hidden_dims) + 1 # number of layers
        self.dtype = dtype
        self.std = wt_init_std
        
        self.D = input_dim
        self.K = output_dim
        self.H = hidden_dims + [output_dim]
        self.params = {}
        
        for ii in range(self.L):
            
            name_W = 'W'+ str(ii+1)
            name_b = 'b'+str(ii+1)
            
            if ii == 0:
                self.params[name_W] = self.std*np.random.randn(self.D,self.H[ii]) # first hidden layer
                self.params[name_b] = np.zeros((self.H[ii],))
            else:
                self.params[name_W] = self.std*np.random.randn(self.H[ii-1],self.H[ii]) # second layer onwards
                self.params[name_b] = np.zeros((self.H[ii],))
           
        for ii in range(self.L-1):
            if self.use_batchnorm:
                name_gamma = 'gamma' + str(ii+1)
                name_beta = 'beta' + str(ii+1)
                self.params[name_gamma] = np.ones((self.H[ii],))
                self.params[name_beta] = np.zeros((self.H[ii],))
   
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self_dropout_param['seed'] = seed
                
        self.bn_params = {}
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for jj in range(self.L-1)]
            
        for key, val in self.params.items():
            self.params[key] = val.astype(self.dtype)   
            
    def loss(self, X, y=None):
        
        X = X.astype(self.dtype)
        if y is None:
            mode = 'test'
        else:
            mode = 'train'
            
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
                   
        
        if self.use_batchnorm and self.use_dropout:
            cache  =  list(np.empty((self.L,)))
            reg_sum = 0.0
            
            # forward pass
            for ii in range(self.L):
                name_W = 'W'+ str(ii+1)
                name_b = 'b'+str(ii+1)
                                
                w = self.params[name_W]
                b = self.params[name_b]
                
                
                if ii == self.L-1:
                    out, cache[ii] = affine_forward(X,w,b)
                    
                else:
                    name_gamma = 'gamma' + str(ii+1)
                    name_beta = 'beta' + str(ii+1)
                    gamma = self.params[name_gamma]
                    beta = self.params[name_beta]
                    out, cache[ii] = affine_bn_relu_dropout_forward(X,w,b, gamma, beta, self.bn_params[ii],                                                                     self.dropout_param)
                    X = out
                
                reg_sum+= 0.5*self.reg*np.sum(w*w) # accumulation of regularization loss
                
                
            scores = out
            if y is None:
                return scores
            else:
                data_loss, dscores = softmax_loss(scores,y)
                loss = data_loss + reg_sum
                
                grads = {}
                dout = dscores
                
                # backward pass
                for ii in range(self.L, 0, -1):
                    name_W = 'W'+ str(ii)
                    name_b = 'b'+str(ii)
                   
                
                    if ii == self.L:
                        
                        dout, dw, db = affine_backward(dout, cache[ii-1])
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                        
                    else:
                        
                        name_gamma = 'gamma' + str(ii)
                        name_beta = 'beta' + str(ii)
                        dtemp, dw, db, dgamma, dbeta = affine_bn_relu_dropout_backward(dout, cache[ii-1])
                        dout = dtemp
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                        grads[name_gamma] = dgamma
                        grads[name_beta] = dbeta
                        
                  
        elif self.use_batchnorm and not self.use_dropout:
            cache  =  list(np.empty((self.L,)))
            reg_sum = 0
            
            # forward pass
            for ii in range(self.L):
                name_W = 'W'+ str(ii+1)
                name_b = 'b'+str(ii+1)
                
                
                w = self.params[name_W]
                b = self.params[name_b]
                
                
                if ii == self.L-1:
                    out, cache[ii] = affine_forward(X,w,b)
                    
                else:
                    name_gamma = 'gamma' + str(ii+1)
                    name_beta = 'beta' + str(ii+1)
                    gamma = self.params[name_gamma]
                    beta = self.params[name_beta]
                    out, cache[ii] = affine_bn_relu_forward(X,w,b, gamma, beta, self.bn_params[ii])
                    X = out
                    
                reg_sum+= 0.5*self.reg*np.sum(w*w) # accumulation of regularization loss
                
                
            scores = out
            if y is None:
                return scores
            else:
                data_loss, dscores = softmax_loss(scores,y)
                loss = data_loss + reg_sum
                
                grads = {}
                dout = dscores
                
                # backward pass
                for ii in range(self.L, 0, -1):
                    name_W = 'W'+ str(ii)
                    name_b = 'b'+str(ii)
                    
                    if ii == self.L:
                        
                        dout, dw, db = affine_backward(dout, cache[ii-1])
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                        
                    else:
                        dtemp, dw, db, dgamma, dbeta = affine_bn_relu_backward(dout, cache[ii-1])
                        dout = dtemp
                        
                        name_gamma = 'gamma' + str(ii)
                        name_beta = 'beta' + str(ii)
                
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                        grads[name_gamma] = dgamma
                        grads[name_beta] = dbeta
  
        
        elif self.use_dropout and not self.use_batchnorm:
            cache  =  list(np.empty((self.L,)))
            reg_sum = 0
            
            # forward pass
            for ii in range(self.L):
                name_W = 'W'+ str(ii+1)
                name_b = 'b'+str(ii+1)
                               
                w = self.params[name_W]
                b = self.params[name_b]
               
                if ii == self.L-1:
                    out, cache[ii] = affine_forward(X,w,b)
                    
                else:
                    out, cache[ii] = affine_relu_dropout_forward(X,w,b, self.dropout_param)
                    X = out
                    
                    
                reg_sum+= 0.5*self.reg*np.sum(w*w) # accumulation of regularization loss
                
                
            scores = out
            if y is None:
                return scores
            else:
                data_loss, dscores = softmax_loss(scores,y)
                loss = data_loss + reg_sum
                
                grads = {}
                dout = dscores
                
                # backward pass
                for ii in range(self.L, 0, -1):
                    name_W = 'W'+ str(ii)
                    name_b = 'b'+str(ii)
                                   
                    if ii == self.L:
                        dout, dw, db = affine_backward(dout, cache[ii-1])
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                       
                    else:
                        dtemp, dw, db = affine_relu_dropout_backward(dout, cache[ii-1])
                        dout = dtemp
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                                        
          
        elif not self.use_dropout and not self.use_batchnorm:
            cache  =  list(np.empty((self.L,)))
            reg_sum = 0
            
            # forward pass
            for ii in range(self.L):
                name_W = 'W'+ str(ii+1)
                name_b = 'b'+str(ii+1)
                               
                w = self.params[name_W]
                b = self.params[name_b]
               
                if ii == self.L-1:
                    out, cache[ii] = affine_forward(X,w,b)
                    
                else:
                    out, cache[ii] = affine_relu_forward(X,w,b)
                    X = out
                      
                reg_sum+= 0.5*self.reg*np.sum(w*w) # accumulation of regularization loss
                
                
            scores = out
            if y is None:
                return scores
            else:
                data_loss, dscores = softmax_loss(scores,y)
                loss = data_loss + reg_sum
                
                grads = {}
                dout = dscores
                
                # backward pass
                for ii in range(self.L, 0, -1):
                    name_W = 'W'+ str(ii)
                    name_b = 'b'+str(ii)
                                   
                    if ii == self.L:
                        dout, dw, db = affine_backward(dout, cache[ii-1])
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
                       
                    else:
                        dtemp, dw, db = affine_relu_backward(dout, cache[ii-1])
                        dout = dtemp
                        
                        grads[name_W] = dw + self.reg*self.params[name_W]
                        grads[name_b] = db + self.reg*self.params[name_b]
        
        return loss, grads
        
    


# In[ ]:



