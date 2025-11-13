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

#  Parámetros de integración
N = 200
delta_t = 0.1
U = zeros((N+1, 4))

# Condiciones iniciales (posición (1,0), velocidad (0,1))
U[0, :] = array([1, 0, 0, 1])

###########################################################
##       EULER
###########################################################

for n in range(0,N):
         
         U[n+1,:] = U[n,:] + delta_t*F(U[n,:])
    
Uee = U
###########################################################
##       Runge kuta
###########################################################
U = zeros((N+1, 4))
U[0, :] = array([1, 0, 0, 1])

for n in range(0, N):
        k1 = F(U[n, :])
        k2 = F(U[n, :] + 0.5 * delta_t * k1)
        k3 = F(U[n, :] + 0.5 * delta_t * k2)
        k4 = F(U[n, :] + delta_t * k3)

        U[n+1, :] = U[n, :] + (delta_t/6.0)*(k1 + 2*k2 + 2*k3 + k4)

Urk= U

###########################################################
##       Crank Nicolson
###########################################################
U = zeros((N+1, 4))
U[0, :] = array([1, 0, 0, 1])
maxiter=50
tol= 1e-10
for n in range(0,N):
        Un1= U[n,:] + delta_t/2*F(U[n,:])
        for i in range(maxiter):
              Un1 = U[n,:] + delta_t/2*(F(U[n,:] )+F(Un1))
              if norm(Un1-U[n,:])> tol:
                    break
    
        U[n+1,:]= U[n,:]+ delta_t/2*(F(U[n,:])+F(Un1))
    
Ucn=U

plt.figure()
plt.axis("equal")

plt.plot(Uee[:, 0], Uee[:, 1], label="Uee")
plt.plot(Urk[:, 0], Urk[:, 1], label="Urk")
plt.plot(Ucn[:, 0], Ucn[:, 1], label="Ucn")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Órbita")
plt.legend()
plt.show()