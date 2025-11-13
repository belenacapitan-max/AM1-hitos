
from numpy import array, concatenate, zeros
from numpy.linalg import norm
import matplotlib.pyplot as plt

def F(U):
    """
    Campo gravitatorio central 2D
    U = [x, y, vx, vy]
    """
    r = U[0:2]       # posición
    rd = U[2:4]      # velocidad
    return concatenate((rd, -r / norm(r)**3), axis=None)

def Euler_explicito(delta_t,U,N):
    
    for n in range(0,N):
         
         U[n+1,:]=U[n,:] + delta_t*F(U[n,:])
    
    return U

# def Euler(U, t1, t2, F):
# return U+ (t2-t1)*F(U,t1)

def Runge_Kutta_4(delta_t, U, N):
    for n in range(0, N):
        k1 = F(U[n, :])
        k2 = F(U[n, :] + 0.5 * delta_t * k1)
        k3 = F(U[n, :] + 0.5 * delta_t * k2)
        k4 = F(U[n, :] + delta_t * k3)

        U[n+1, :] = U[n, :] + (delta_t/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    return U

def Crank_Nicolson_step(delta_t, U, maxiter=50, tol= 1e-10):
    Un1= U + delta_t/2*F(U)
    for i in range(maxiter):
        Un1 = U + delta_t/2*(F(U)+F(Un1))
        if norm(Un1-U)> tol:
            break
    
    return Un1

def Crank_Nicolson(delta_t, U, N):

    for n in range(0,N):
        Un1=Crank_Nicolson_step(delta_t, U[n,:])
        U[n+1,:]= U[n,:]+ delta_t/2*(F(U[n,:])+F(Un1))
    
    return U

print('EE=Euler Explicito')
print('RK4=Runge Kutta 4')
print('CN= crank Nicolson con interpolacion lineal')
method=input('Escoge un metodo de integracion (EE, RK4, CN o todos):')

# Parámetros de integración
N = 200
delta_t = 0.1
U = zeros((N+1, 4))

# Condiciones iniciales (posición (1,0), velocidad (0,1))
U[0, :] = array([1, 0, 0, 1])
if method=='EE':
    U_ee= Euler_explicito(delta_t, U, N)

    plt.figure()
    plt.axis("equal")
    plt.plot(U_ee[:, 0], U_ee[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Órbita con integración Euler E")
    plt.show()
elif method=='RK4':
    U_rk = Runge_Kutta_4(delta_t, U, N)

    plt.figure()
    plt.axis("equal")
    plt.plot(U_rk[:, 0], U_rk[:, 1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Órbita con integración RK4")
    plt.show()
elif method=='CN':
    U_CN = Crank_Nicolson(delta_t, U, N)

    plt.figure()
    plt.axis("equal")
    plt.plot(U_CN[:,0],U_CN[:,1])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Órbita con integración Crank N")
    plt.show()
elif method=='todos':
    U_EE= Euler_explicito(delta_t, U, N)
    U_rk = Runge_Kutta_4(delta_t, U, N)
    U_CN = Crank_Nicolson(delta_t, U, N)

    plt.figure()
    plt.axis("equal")
    plt.plot(U_CN[:,0],U_CN[:,1], label='CN')
    plt.plot(U_rk[:, 0], U_rk[:, 1], label='RK4')
    plt.plot(U_EE[:, 0], U_EE[:, 1], label='EE')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Órbita con integración EE, RK4, CN")
    plt.legend
    plt.show()
else:
    print('Input no valido, prueba otra vez')