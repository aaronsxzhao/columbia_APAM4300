import numpy 
from  scipy.special import factorial

def fdcoeffV(k,xbar,x):
    """
    fdcoeffV routine modified from Leveque (2007) matlab function
    
    Params:
    -------
    
    k: int
        order of derivative
    xbar: float
        point at which derivative is to be evaluated
    x: ndarray
        numpy array of coordinates to use in calculating the weights
    
    Returns:
    --------
    c: ndarray
        array of floats of coefficients.  

    Compute coefficients for finite difference approximation for the
    derivative of order k at xbar based on grid values at points in x.

    WARNING: This approach is numerically unstable for large values of n since
    the Vandermonde matrix is poorly conditioned.  Use fdcoeffF.m instead,
    which is based on Fornberg's method.

     This function returns a row vector c of dimension 1 by n, where n=length(x),
     containing coefficients to approximate u^{(k)}(xbar), 
     the k'th derivative of u evaluated at xbar,  based on n values
     of u at x(1), x(2), ... x(n).  

     If U is an array containing u(x) at these n points, then 
     c.dot(U) will give the approximation to u^{(k)}(xbar).

     Note for k=0 this can be used to evaluate the interpolating polynomial 
     itself.

    Requires len(x) > k.  
    Usually the elements x(i) are monotonically increasing
    and x(1) <= xbar <= x(n), but neither condition is required.
    The x values need not be equally spaced but must be distinct.  
    
    Modified rom  http://www.amath.washington.edu/~rjl/fdmbook/  (2007)
    """
    

    n = x.shape[0]
    assert  k < n, " The order of the derivative must be less than the stencil width"

    # Generate the Vandermonde matrix from the Taylor series
    A = numpy.ones((n,n))
    xrow = (x - xbar)  # displacements x-xbar 
    for i in range(1,n):
        A[i,:] = (xrow**(i))/factorial(i);
        
    b = numpy.zeros(n)    # b is right hand side,
    b[k] = 1              # so k'th derivative term remains

    c = numpy.linalg.solve(A,b)          # solve n by n system for coefficients
    
    return c
