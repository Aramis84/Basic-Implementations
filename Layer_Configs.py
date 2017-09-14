from Layers import *

"""
Implements forward and backward passes of some common layer configurations
"""

def affine_relu_forward(X,w,b):
    """
    affine -> relu
    """
    aff, aff_cache =  affine_forward(X,w,b)
    out, relu_cache = relu_forward(aff)
    
    cache = (aff_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout,cache):
    """
    Backward pass for
    affine -> relu
    """
    aff_cache, relu_cache =  cache
    daff = relu_backward(dout, relu_cache)
    dX, dw, db = affine_backward(daff, aff_cache)
    
    return dX, dw, db

def affine_bn_relu_forward(X,w,b, gamma, beta, bn_param):
    """
    affine -> batch normalization -> relu
    """
    
    aff, aff_cache =  affine_forward(X,w,b)
    bn, bn_cache = batchnorm_forward(aff, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    
    cache = (aff_cache, bn_cache, relu_cache)
    return out, cache

def affine_bn_relu_backward(dout,cache):
    """
    Backward pass for
    affine -> batch normalization -> relu
    """
    aff_cache, bn_cache, relu_cache =  cache
    dbn = relu_backward(dout, relu_cache)
    daff, dgamma, dbeta  = batchnorm_backward(dbn, bn_cache)
    dX, dw, db = affine_backward(daff, aff_cache)
    
    return dX, dw, db, dgamma, dbeta

def affine_relu_dropout_forward(X,w,b, dropout_param):
    """
    affine -> relu -> dropout
    """
    aff, aff_cache =  affine_forward(X,w,b)
    relu, relu_cache = relu_forward(aff)
    out, dropput_cache = dropout_forward(relu, dropout_param)
    
    cache = (aff_cache, relu_cache, dropout_cache)
    return out, cache


def affine_relu_dropout_backward(dout, cache):
    """
    Backward pass for
    affine -> relu -> dropout
    """
    aff_cache, relu_cache, dropout_cache =  cache 
    ddropout = dropout_backward(dout, dropout_cache)
    drelu = relu_backward(ddropout, relu_cache)
    dX, dw, db = affine_backward(drelu, aff_cache)
    
    return dX, dw, db


def affine_bn_relu_dropout_forward(X,w,b, gamma, beta, bn_param, dropout_param):
    """
    affine -> batch normalization -> relu -> dropout
    """
    
    aff, aff_cache =  affine_forward(X,w,b)
    bn, bn_cache = batchnorm_forward(aff, gamma, beta, bn_param)
    relu, relu_cache = relu_forward(bn)
    out, dropput_cache = dropout_forward(relu, dropout_param)
        
    cache = (aff_cache, bn_cache, relu_cache, dropput_cache)
    return out, cache

def affine_bn_relu_dropout_backward(dout, cache):
    """
    Backward pass for
    affine -> batch normalization -> relu -> dropout
    """
    aff_cache, bn_cache, relu_cache, dropout_cache =  cache
    ddropout = dropout_backward(dout, dropout_cache)
    dbn = relu_backward(ddropout, relu_cache)
    daff, dgamma, dbeta  = batchnorm_backward(dbn, bn_cache)
    dX, dw, db = affine_backward(daff, aff_cache)
    
    return dX, dw, db, dgamma, dbeta

