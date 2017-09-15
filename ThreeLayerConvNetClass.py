import numpy as np
from ConvNetLayers import *
from ConvNetLayer_Configs import *
from Layer_Configs import *
from Layers import *

class ThreeLayerConvNet(object):
    """
    A three-layer convolutional neural network with ReLU nonlinearity and
    softmax loss. Input data set has (N x C x H x W) dimensions and classification is performed over K classes.
    The layers are
    
    conv -> relu -> max pool -> full connected (1 hidden layer with ReLu) -> output layer

    This class does not implement an optimizer. It will interact with a separate solver (provided in the CS 231n class material)
    
    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_filters, filter_size = (7,), pool_kernel = (2,), pad = 1,                                      filter_stride = 1, pool_stride =2,  wt_init_std = 0.001, reg = 0.0, dtype = np.float32):
        
        self.reg = reg
        self.dtype = dtype
        self.std = wt_init_std
        self.C, self.H, self.W = input_dim
        
        if len(filter_size)==1:
            self.HH = self.WW = filter_size[0]
        else:
            self.HH = filter_size[0]
            self.WW = filter_s1ze[1]
            
        if len(pool_kernel)==1:
            self.poolh = self.poolw = pool_kernel[0]
        else:
            self.poolh = pool_kernel[0]
            self.poolw = pool_kernel[1]
                       
        self.F = num_filters
        self.pad = pad
        self.fstride = filter_stride
        self.pstride = pool_stride
        self.hidden = hidden_dim
        self.K = output_dim
        
        # checking to see if filter fits properly in the padded/nonpadded image space
        var_H_conv = (self.H - self.HH +2*self.pad)%self.fstride
        var_W_conv = (self.W - self.WW +2*self.pad)%self.fstride
    
        assert (var_H_conv == 0 and var_W_conv ==0), 'Filter and input dimensions do not work. Change padding or filter dimensions'
         
        dim_H_conv = int((self.H - self.HH +2*self.pad)/self.fstride) + 1
        dim_W_conv = int((self.W - self.WW +2*self.pad)/self.fstride) + 1
        
        var_H_pool = (dim_H_conv - self.poolh)%self.pstride
        var_W_pool = (dim_W_conv - self.poolw)%self.pstride
        
        assert (var_H_pool == 0 and var_W_pool ==0), 'Pool kernel and input dimensions do not work. Change filterdimensions and/or pool stride' 
      
        dim_H_pool = int((dim_H_conv - self.poolh)/self.pstride) + 1
        dim_W_pool = int((dim_W_conv - self.poolw)/self.pstride) + 1

        
        self.W1 = self.std*np.random.randn(self.F, self.C, self.HH, self.WW)
        self.b1 = np.zeros((self.F,))
        
        self.D = dim_H_pool*dim_W_pool*self.F
        
        self.W2 = self.std*np.random.randn(self.D,self.hidden)
        self.b2 = np.zeros((self.hidden,))
        
        self.W3 = self.std*np.random.randn(self.hidden, self.K)
        self.b3 = np.zeros((self.K,))
        
        self.params = {}
        self.params['W1'] = self.W1
        self.params['b1'] = self.b1 
        self.params['W2'] = self.W2
        self.params['b2'] = self.b2
        self.params['W3'] = self.W3
        self.params['b3'] = self.b3
        
        for key, val in self.params.items():
            self.params[key] = val.astype(self.dtype)   
        
        
    def loss(self, X, y = None):
        
        X = X.astype(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        
        # parameters for forward pass of convolutional layer
        conv_param = {'stride': self.fstride, 'pad': self.pad}

        # parameters for forward pass of max-pooling layer
        pool_param = {'pool_height': self.poolh, 'pool_width': self.poolw, 'stride': self.pstride}
        
        # forward pass
        out1, cache1 = conv_relu_forward(X,W1,b1, conv_param)
        out2, cache2 = max_pooling_forward(out1, pool_param)
        out3, cache3 = affine_relu_forward(out2, W2, b2)
        out4, cache4 = affine_forward(out3, W3, b3)
        
        scores = out4
        
        if y is None:
            return scores
        else:
            data_loss, dscores = softmax_loss(scores,y)
            reg_loss_W1 = 0.5*self.reg*np.sum(W1*W1)
            reg_loss_W2 = 0.5*self.reg*np.sum(W2*W2)
            reg_loss_W3 = 0.5*self.reg*np.sum(W3*W3)
            loss = data_loss + reg_loss_W1 + reg_loss_W2 
            
            #backward pass
            dout4, dW3, db3 = affine_backward(dscores, cache4)
            dout3, dW2, db2 = affine_relu_backward(dout4,cache3)
            dout2 = max_pooling_backward(dout3, cache2)
            dout1, dW1, db1 = conv_relu_backward(dout2,cache1)
            
            reg_grad_W1 = self.reg*W1
            reg_grad_W2 = self.reg*W2
            reg_grad_W3 = self.reg*W3
            
            grads = {}
            grads['W1'] = dW1 + reg_grad_W1
            grads['b1'] = db1
            grads['W2'] = dW2 + reg_grad_W2
            grads['b2'] = db2
            grads['W3'] = dW3 + reg_grad_W3
            grads['b3'] = db3
            
            return loss, grads     

