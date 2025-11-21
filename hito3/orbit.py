from numpy import concatenate
from numpy.linalg import norm

def F(U):
    ''' Entra una función U que contiene el vector posición y el vector velocidad. 
    Y devuelve un vector F derivada de U respecto al tiempo, para una masa y mu = 0
    '''
    r = U[0:2]
    rdot = U[2:4]
    return concatenate ((rdot, -r/norm(r)**3), axis=None)