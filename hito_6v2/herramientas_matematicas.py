from numpy import zeros
from numpy.linalg import solve, norm


def derivative(f, x, dx):
    ''' Calcula la derivada de un vector
    '''
    h = 1e-7 # dx escalar
    return (f(x + dx) - f(x - dx)) / (2 * h)

def Jacobian(f, x):
    J = zeros((len(x), len(x)))
    for j in range(len(x)):
        dx = zeros(len(x))
        dx[j] = 1e-7
        J[:,j] = derivative(f, x, dx)

    return J
    
def Gauss(A, b):
    return solve(A, b)

def Newton(f, x0):
    x = x0
    Dx = 1.0
    while norm(Dx) > 1e-10:
        A = Jacobian(f, x)
        Dx = Gauss(A, -f(x))
        x = x + Dx
    
    return x
