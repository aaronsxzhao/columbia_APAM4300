import numpy
import matplotlib.pyplot as plt

def unit_ball_image(axes, A, ord, title=True):
    """Plot  the unit ball and its image under the tranformation resulting from *A* 
    and return an estimate of the matrix norm 
    
    :Input:
     - *axes* (matplotlib.axes) Axes to plot on
     - *A* (ndarray) Matrix that represents the mapping (transformation)
     - *ord* (float) The norm requested.
     
    :Output:
     - (float) Maximum norm estimate (i.e. the matrix norm)
    """    
    # construct the unit ball and extract points on the 1-contour
    N = 100
    X, Y = numpy.meshgrid(numpy.linspace(-1.1, 1.1, N), numpy.linspace(-1.1, 1.1, N))
    V = numpy.empty((N, N))
    for i in range(N):
        for j in range(N ):
            V[i, j] = numpy.linalg.norm(numpy.array([X[i, j], Y[i, j]]), ord=ord)
    contourset = axes.contour(X, Y, V, 'w', levels=[1])
    
    # nifty matplotlib trick for extracting the coordinates of the contour
    ball = contourset.allsegs[0][0]
    axes.plot(ball[:,0], ball[:,1],'b')
    
    # calculate the image of the unit ball under A and plot it
    image = A.dot(ball.T).T
    axes.plot(image[:,0],image[:,1],'r')

    # estimate the norm
    max_norm = numpy.max(numpy.linalg.norm(image, ord=ord, axis=1))
    
    #prettify
    axes.grid()
    axes.legend(['$||\mathbf{{x}}||_{}=1$'.format(ord),
             '$||A\\mathbf{{x}}||_{}$'.format(ord)], loc='best')
    
    if title:
        axes.set_title('$||A||_{}\\approx{:3.5f}$'.format(ord,max_norm))
   
    return max_norm
    