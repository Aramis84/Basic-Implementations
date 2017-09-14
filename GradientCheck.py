import numpy as np
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


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        # evaluate function at x+h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h) # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext() # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad



# In[ ]:



