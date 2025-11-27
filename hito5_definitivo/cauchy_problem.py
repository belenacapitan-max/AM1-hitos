from numpy import zeros, linspace, array
from  orbit import F
from temporal_schemes import Crank_Nicolson
import matplotlib.pyplot as plt

def Cauchy_problem(F, U0, t, Temporal_scheme):
    N = len(t) - 1
    Nv = len(U0) # Numero de variable
    U = zeros((N+1, Nv))
    U[0, :]= U0

    for n in range(N):
        U[n+1, :] = Temporal_scheme(U[n,:], t[n], t[n+1], F)
    return U 

def Test_Cauchy():
    # Variables
    U0 = array([1, 0, 0, 1])
    T = 200
    N = 10000
    t = linspace(0, T, N + 1)

    # Soluciones 
    U_CN = Cauchy_problem(F, U0, t, Crank_Nicolson)

    # Gráfico comparativo 
    plt.plot(U_CN[:,0], U_CN[:,1], color='green', label='Crank-Nicolson')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Trayectorias en el plano")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()