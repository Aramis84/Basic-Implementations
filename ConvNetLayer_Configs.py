from ConvNetLayers import *
from Layers import *

"""
Implements forward and backward passes of some common CNN layer configurations
"""

def conv_relu_forward(X,w,b, conv_param):
    """
    conv -> relu 
    """
    conv, conv_cache =  conv_forward(X, w, b, conv_param)
    out, relu_cache = relu_forward(conv)
     
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout,cache):
    """
    Backward pass for
    conv -> relu 
    """
    conv_cache, relu_cache =  cache
    dconv = relu_backward(dout, relu_cache)
    dX, dw, db = conv_backward(dconv, conv_cache)
    
    return dX, dw, db


def conv_relu_pool_forward(X,w,b, conv_param, pool_param):
    """
    conv -> relu -> pool
    """
    conv, conv_cache =  conv_forward(X, w, b, conv_param)
    relu, relu_cache = relu_forward(conv)
    out, pool_cache = max_pooling_forward(relu, pool_param)
    
    
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout,cache):
    """
    Backward pass for
    conv -> relu -> pool
    """
    conv_cache, relu_cache, pool_cache =  cache
    drelu = max_pooling_backward(dout, pool_cache)
    dconv = relu_backward(drelu, relu_cache)
    dX, dw, db = conv_backward(dconv, conv_cache)
    
    return dX, dw, db

