
# coding: utf-8

# In[89]:

import numpy as np

"""
Implentations of convolutional and max pooling layers. These are not vectorized implementations

"""

def conv_forward(X, w, b, conv_param):
    """
    Implements forward pass of a convolution layer
    Inputs :X input matrix -> N x C x H x W (N samples each with C channels, and spatial size HxW)
           w filter weights -> F x C x HH x WW (F filters each with C channels (must be same as input channels), and 
           spatial size HHxWW)
           b bias -> bias terms for each filter (1D array with F elements)
           conv_param -> dictionary of hyperparameters
               stride : stride length
               pad : number of paddings , 0 means no padding is used
               
    Outputs :out -> output of the convolutional layer N x F x dim_H x dim_w
            cache -> collection of variables required in the backward pass
    """
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = X.shape
    F, _, HH, WW = w.shape 
    
    # checking to see if filter fits properly in the padded/nonpadded image space
    var_H = (H - HH +2*pad)%stride
    var_W = (W - WW +2*pad)%stride
    
    if var_H != 0 or var_W !=0:
        print('Filter and input dimensions do not work. Change padding or filter dimensions')
    else:
        dim_H = int((H - HH +2*pad)/stride) + 1
        dim_W = int((W - WW +2*pad)/stride) + 1
        result = np.zeros((N, F, dim_H, dim_W))
        
        all_patch = np.zeros((N,F,dim_H, dim_W,C,HH,WW))
        for ii in range(N):
            x_pad = np.pad(X[ii,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant') # padding is added only along height and
            # width axes
            for f in range(F):
                filt = w[f,:,:,:]
                for height in range(dim_H):
                    for width in range(dim_W):
                        tlc = height*stride # top left corner of the patch
                        trc = width*stride # top right corner of the patch
                        blc = tlc + HH # bottom left corner of the patch
                        brc = trc + WW # bottom right corner of the patch
                        
                        patch = x_pad[:,tlc:blc, trc:brc] # extracting patch from the image
                        result[ii,f,height,width] = np.sum(patch*filt) + b[f]
                        
                        all_patch[ii,f,height,width,:,:,:] = patch # collecting the image patches as these are part of the
                        # gradients of the convolution layer output with repsect to the filter weights
                 
                

    out = result
    cache = (X,w,b, conv_param, all_patch)
    return out, cache



def conv_backward(dout, cache):
    """
    Implements backward pass of a convolution layer.

    Inputs :dout -> Upstream derivatives (N x F x dim_H x dim_w)
            cache-> collection of variables required in the backward pass
    
    Ouputs : dX: Gradient with respect to X (N x C x H x W)
             dw: Gradient with respect to w (F x C x HH x WW)
             db: Gradient with respect to b (1D array with F elements)
    """
    
    X,w,b, conv_param, all_patch = cache
    N, C, H, W = X.shape
    F, _, HH, WW = w.shape 
    _, _, dim_H, dim_W = dout.shape
    
    
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    dX = np.zeros(X.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    
    for ii in range(N):
        x_pad = np.pad(X[ii,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
        dX_pad = np.pad(dX[ii,:,:,:], ((0,0),(pad,pad),(pad,pad)), 'constant')
        
        for f in range(F): 
            filt = w[f,:,:,:]
            for height in range(dim_H):
                for width in range(dim_W):
                    dw[f,:,:,:] += all_patch[ii,f,height,width,:,:,:]*dout[ii,f,height,width]
                    db[f] += dout[ii,f,height,width]
                    
                    tlc = height*stride # top left corner of the patch
                    trc = width*stride # top right corner of the patch
                    blc = tlc + HH # bottom left corner of the patch
                    brc = trc + WW 
                    
                    dX_pad[:,tlc:blc, trc:brc] += filt*dout[ii,f,height,width] # summing over all filters 
        dX[ii,:,:,:] = dX_pad[:,1:-1, 1:-1] # leaving out the derivatives with respect to padded values            
    
    return dX, dw, db



def max_pooling_forward(X, pool_param):
    """
    Implements forward pass of a max pooling layer
    Inputs :X input matrix -> N x F x H x W (N samples each with F channels, and spatial size HxW, coming in from a 
                              previous convolutional layer)
            pool_param -> dictionary of hyperparameters
               stride : stride length
               pool_height : number of rows in pool filter
               pool_width : number of columns in pool filter
               
    Outputs :out -> output of the convolutional layer N x F x dim_H x dim_w
            cache -> collection of variables required in the backward pass
    """
      
    stride = pool_param['stride']
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    N, F, H, W = X.shape
    
    var_H = (H - ph)%stride
    var_W = (W -pw)%stride
    
    if var_H != 0 or var_W !=0:
        print('Filter and input dimensions do not work. Change padding or filter dimensions')
    else:
        dim_H = int((H - ph)/stride) + 1
        dim_W = int((W - pw)/stride) + 1
        result = np.zeros((N,F, dim_H, dim_W))
    
        all_max_pos = np.zeros((N,F,dim_H, dim_W))
        for ii in range(N):
            for f in range(F):
                for height in range(dim_H):
                    for width in range(dim_W):                    
                        tlc = height*stride # top left corner of the patch
                        trc = width*stride # top right corner of the patch
                        blc = tlc + ph # bottom left corner of the patch
                        brc = trc + pw # bottom right corner of the patch

                        patch = X[ii, f, tlc:blc, trc:brc] # extracting patch from the image
                        result[ii,f,height,width] = np.max(patch)
                            
                        all_max_pos[ii,f,height,width] = np.argmax(patch) # collecting the index of the maximum
                        
    out = result
    cache = (X, pool_param, all_max_pos)
    return out, cache



def max_pooling_backward(dout, cache):
    """
    Implements backward pass of a max pooling layer.

    Inputs :dout -> Upstream derivatives (N x F x dim_H x dim_w)
            cache-> collection of variables required in the backward pass
    
    Ouputs : dX: Gradient with respect to X (N x F x H x W)
          
    """
    
    X, pool_param, all_max_pos = cache
    N, F, H, W = X.shape
    _, _, dim_H, dim_W = dout.shape
    
    
    stride = pool_param['stride']
    ph = pool_param['pool_height']
    pw = pool_param['pool_width']
    
    dX = np.zeros(X.shape)
    for ii in range(N):
        for f in range(F): 
            for height in range(dim_H):
                for width in range(dim_W):
                    tlc = height*stride # top left corner of the patch
                    trc = width*stride # top right corner of the patch
                    blc = tlc + ph # bottom left corner of the patch
                    brc = trc + pw # bottom right corner of the patch
                    
                    pos = int(all_max_pos[ii,f,height,width])
                    
                    patch = X[ii, f, tlc:blc, trc:brc]
                    patch1 = np.reshape(patch, (ph*pw))
                    patch1 = np.zeros(patch1.shape)
                    patch1[pos] = 1
                    patch1 = patch1.reshape(patch.shape)
                    
                    dX[ii,f, tlc:blc, trc:brc] = patch1*dout[ii,f,height,width] 
  
    return dX

