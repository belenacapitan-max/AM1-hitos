from numpy import  array,  sqrt
from numpy.linalg import eig
from three_body_problem import CR3BP
from herramientas_matematicas import Newton, Jacobian

def F_equilibrio(X, mu):
    """
    X = [x, y]
    Devuelve F(X) = [ax(x,y), ay(x,y)] imponiendo z=0, vx=vy=vz=0.
    """
    x, y = X

    U = array([x, y, 0.0, 0.0, 0.0, 0.0])
    dU_dt = CR3BP(0.0, U, mu)
    ax = dU_dt[3]
    ay = dU_dt[4]

    return array([ax, ay])

def initial_guesses(mu):
    """
    Devuelve iniciales aproximados para L1..L5.
    """
    # Aproximaciones para los colineales (x sobre el eje, y=0)
    # L1 entre las masas
    xL1 = 1 - (mu/3.0)**(1.0/3.0)
    # L2 a la derecha del cuerpo pequeño
    xL2 = 1 + (mu/3.0)**(1.0/3.0)
    # L3 a la izquierda del cuerpo grande
    xL3 = -1 - (5.0*mu/12.0)  # aproximación clásica

    # Triangulares (casi vértices de triángulo equilátero)
    xL4 = 0.5 - mu
    yL4 =  sqrt(3.0)/2.0
    xL5 = 0.5 - mu
    yL5 = -sqrt(3.0)/2.0

    guesses = {
        "L1": array([xL1, 0.0]),
        "L2": array([xL2, 0.0]),
        "L3": array([xL3, 0.0]),
        "L4": array([xL4, yL4]),
        "L5": array([xL5, yL5]),
    }
    return guesses

def find_lagrange_points(mu, tol=1e-12):
    pts = {}
    guesses = initial_guesses(mu)
    for name, X0 in guesses.items():
        f = lambda X: F_equilibrio(X, mu)
        X_star = Newton(f, X0)
        pts[name] = X_star
    return pts



def Estabilidad(U_eq, mu, tol):

    def F(U):
        return CR3BP(0.0, U, mu)

    J = Jacobian(F, U_eq) # Jacobiano particulaarizado en los puntos de equilibrio

    vals, vect = eig(J)
    
    real = vals.real

    estable = (real.max() <= tol)

    tipo = Clasificacion_estabilidad(vals, tol)

    return estable, tipo, vals

def Clasificacion_estabilidad(vals, tol):

    real = vals.real 
    
    max_real = real.max()
    min_real = real.min()
    
    if max_real > tol:

        tipo = "Inestable"

    elif min_real < -tol:

        tipo = "Asintoticamente Estable"

    else :

        tipo = "Marginalmente Estable"

    return tipo

mu = 0.0121505856
pts = find_lagrange_points(mu)
tol = 1e-8

for name, X in pts.items():
    x_star, y_star = X
    U_eq = array([x_star, y_star, 0.0, 0.0, 0.0, 0.0])

    estable, tipo, vals = Estabilidad(U_eq, mu, tol)

    # print(f"{name}: estable = {estable}, tipo = {tipo}")
    # print("  autovalores:", vals)
    # print("  partes reales:", vals.real)
    # print()