
from __future__ import division
import numpy as np
"""
Implements various first order update rules commonly used for training neural networks. Each rule accepts the gradient of the 
loss with respect to the parameters, the current parameters. Each function also has a configuration dictionary for storing 
the default hyperparameter values which can be overridden if performing cross-validation. The functions return the updated 
parameter vector and the updated configuration dict for use in the next iteration. The intermediate values that need to be cached
for some update rules are also stored in the config dictionary.
"""


# Stochastic Gradient Descent
def sgd(w, dw, config = None):
    """
    Standard stochastic gradient
    config format:
    - learning_rate: scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate',1e-2) # setting default learning rate
    
    eta = config['learning_rate'] 
    next_w = w - eta*dw
    
    return next_w, config


# Stochastic gradient descent with momentum
def sgd_momentum(w, dw, config = None):
    """
    Stochastic gradient using momentum vector
    config format:
    - learning_rate: scalar learning rate.
    - momentum: scalar between 0 and 1, generally representing degree of friction, closer to 1 (no friction) and vice versa.
    - cache: velocity vector: moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9) # set the default "friction coefficient"
    config.setdefault('cache', np.zeros(w.shape)) # initialize the velocity vector
        
    v = config['cache']
    m = config['momentum'] 
    eta = config['learning_rate'] 
    
    v = m*v - eta*dw
    next_w = w + v
    
    config['cache'] = v # update velocity vector, crucial
    
    return next_w, config


# Nesterov Accelerated Gradient
def nesterov_momentum(w, dw, config = None):
    """
   Nesterov accelerated gradient using momentum vector
    config format:
    - learning_rate: scalar learning rate.
    - momentum: scalar between 0 and 1, generally representing degree of friction, closer to 1 (no friction) and vice versa.
    - cache: velocity vector: moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9) # set the default "friction coefficient"
    config.setdefault('cache', np.zeros(w.shape)) # initialize the velocity vector
        
    v = config['cache']
    m = config['momentum'] 
    eta = config['learning_rate'] 
    
    v_prev = v # storing the old v
    v = m*v - eta*dw 
    next_w = w - m*v_prev + (1+m)*v
    
    config['cache'] = v # update velocity vector, crucial
    
    return next_w, config
    
# Adagrad
def adagrad(w, dw, config = None):
    """
    Adaptive learning rate aka AdaGrad
    config format:
    - learning_rate: scalar learning rate.
    - eps: positive scalar used to avoid dividing by zero.
    - cache: accumulator of squared gradient.    
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('eps', 1e-9)
    config.setdefault('cache', np.zeros(w.shape))
        
    s = config['cache']
    eta = config['learning_rate'] 
    eps = config['eps']
    
    s = s + dw**2
    next_w = w - eta*dw/(np.sqrt(s) + eps)
    
    config['cache'] = s # update cache, crucial
    
    return next_w, config

# RMSProp
def rmsprop(w, dw, config = None):
    """
    Modified AdaGrad aka RMSProp
    config format:
    - learning_rate: scalar learning rate.
    - decay_rate: scalar between 0 and 1: decay rate for moving average of squared gradient.
    - eps: positive scalar used to avoid dividing by zero.
    - cache: moving average of squared gradient.       
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('eps', 1e-9)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('cache', np.zeros(w.shape))
        
    s = config['cache']
    eta = config['learning_rate'] 
    eps = config['eps']
    beta = config['decay_rate'] 
    
    s = beta*s + (1-beta)*(dw**2)
    next_w = w - eta*dw/(np.sqrt(s) + eps)
    
    config['cache'] = s # update cache, crucial
    
    return next_w, config

# Adam
def adam(w, dw, config = None):
    """
    Adaptive Moment Estimation aka Adam
    config format:
    - learning_rate: scalar learning rate.
    - beta1: scalar between 0 and 1: decay rate for moving average of gradient.
    - beta2: scalar between 0 and 1: decay rate for moving average of squared gradient.
    - eps: positive scalar used to avoid dividing by zero.
    - m: moving average of gradient.
    - s: moving average of squared gradient.
    - iter: iteration number.    
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('eps', 1e-8)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('iter', 1)
    config.setdefault('m', np.zeros(w.shape))
    config.setdefault('v', np.zeros(w.shape))
        
    m = config['m']
    v = config['v']
    eta = config['learning_rate'] 
    eps = config['eps']
    beta1 = config['beta1'] 
    beta2 = config['beta2'] 
    t = config['iter'] 
    
    m = beta1*m + (1-beta1)*dw
    v = beta2*v + (1-beta2)*(dw**2)
    m_ub = m/(1-beta1**t)
    v_ub = v/(1-beta2**t)
        
    next_w = w - eta*m_ub/(np.sqrt(v_ub) + eps)
    
    # update cache, crucial
    # cache is not updated by bias corrected values but the original moving averaged values. See Adam paper, page 2, algorithm 1.
    
    config['m'] = m 
    config['v'] = v
    config['iter'] = t+1
    
    return next_w, config

