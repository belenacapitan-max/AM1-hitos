from numpy import zeros, linspace, log, polyfit, array
from numpy.linalg import norm
import matplotlib.pyplot as plt
from cauchy_problem import Cauchy_problem
from orbit import F
from temporal_schemes import Euler_explicito, Euler_implicito, RK4, Crank_Nicolson


def Cauchy_Error(F, U0, t, Temporal_scheme,q):
    """
    Estima el error numérico de la solución del problema de Cauchy, usando empleando multimalla.  METODO DE RICHARDSON

    Parámetros
    ------------
    F : función
        F(U) = dU/dt
    U0 : array
        condición inicial
    t : array
        Malla temporal gruesa (N+1)
    Temporal_scheme: función
        Esquema temporal, del tipo Temporal_scheme(U_n, t_n, t_{n+1}, F)
    q : int
        orden teórico del esquema temporal 

    Devuelve
    ------------
    U1 : array --> solución numérica en la malla t (gruesa)
    E : array --> Estimación del error en cada nodo de t (mismas dimensiones que U1)
    """
    N = len(t) - 1 # número de pasos en la malla gruesa
    t1 = t
    #malla fina con 2*N+1 puntos
    t2 = linspace(t[0], t[-1], 2*N + 1) # t[-1] es el último elemento del vector

    #Soluciones de ambas mallas
    U1 = Cauchy_problem(F, U0, t1, Temporal_scheme) # malla gruesaa
    U2 = Cauchy_problem(F, U0, t2, Temporal_scheme) # malla fina

    Nv = len(U0) # Nº de variable x, y, vx, vy ....
    E = zeros((N + 1, Nv)) 

    #Para cada instante de la malla gruesa (t1[n]) lo comparamos con el correspondiente de la malla fina (t2([2*n]))
    # en este caso como la malla gruesa tiene el doble de puntos que la fina coinciden en los puntos pares

    for n in range(N+1):
        E[n, :] = (U1[n, :] - U2[2*n, :]) / (2**q - 1)

    return U1, E

def convergence_rate(F, U0, T, Temporal_scheme, q, N_list):
    """
    Calcular la convergencia de un esquema temporal.

    Parámetros -----------------------------------------
    F : función
        F(U) = dU/dt
    U0 : array
        condición inicial
    t : array
        Malla temporal gruesa (N+1)
    
    Devuelve -------------------------------------------
    p: float
        Pendiente de log(Error) vs log(dt), aproximación de la orden de convergencia
    dt_list : lista de dt usados
    err_list : lista de errores globales correspondientes
    """
    dt_list = []
    err_list = []

    for N in N_list:
        t = linspace(0, T, N+1)
        U_num , E = Cauchy_Error(F, U0, t, Temporal_scheme, q)
        dt = T / N
        dt_list.append(dt)

        # Error global en el tiempo final: norma del vector E(T)
        err_T = norm(E[-1, :])
        err_list.append(err_T)

    # Ajuste lineal en escala log-log
    log_dt = log(dt_list)
    log_err = log(err_list)
    # polyfit devuelve [pendiente, ordenada]
    slope, intercept = polyfit(log_dt, log_err, 1)

    return slope, dt_list, err_list

U0 = array([1, 0, 0, 1])
T = 10.0
N = 2000
t = linspace(0, T, N+1)

# Euler explícito
U_euler, E_euler = Cauchy_Error(F, U0, t, Euler_explicito, q=1)

# Euler implícito
#U_ie, E_ie = Cauchy_Error(F, U0, t, Euler_implicito, q=1)

# Crank–Nicolson
U_cn, E_cn = Cauchy_Error(F, U0, t, Crank_Nicolson, q=2)

# RK4
U_rk4, E_rk4 = Cauchy_Error(F, U0, t, RK4, q=4)

N_list = [400, 800, 1600, 3200]

p_euler, dt_euler, err_euler = convergence_rate(F, U0, T, Euler_explicito, q=1, N_list=N_list)
#p_ie,    dt_ie,    err_ie    = convergence_rate(F, U0, T, Euler_implicito, q=1, N_list=N_list)
p_cn,    dt_cn,    err_cn    = convergence_rate(F, U0, T, Crank_Nicolson, q=2, N_list=N_list)
p_rk4,   dt_rk4,   err_rk4   = convergence_rate(F, U0, T, RK4, q=4, N_list=N_list)

print("Orden numérico Euler explícito     ≈", p_euler)
#print("Orden numérico Euler implícito     ≈", p_ie)
print("Orden numérico Crank_Nicolson      ≈", p_cn)
print("Orden numérico Runge_Kutta 4º      ≈", p_rk4)
#--------------------------------------------------------------------------------------#
# Condición inicial de tu órbita
U0 = array([1.0, 0.0, 0.0, 1.0])
T = 10.0  # tiempo final (puedes poner 200 si quieres)
N = 2000  # número de pasos para visualizar la órbita

# 1) Solución con Crank–Nicolson
t = linspace(0, T, N+1)
U_CN = Cauchy_problem(F, U0, t, Crank_Nicolson)

# 2) Estimación de error para Crank–Nicolson (q=2)
U_CN_coarse, E_CN = Cauchy_Error(F, U0, t, Crank_Nicolson, q=2)

# 3) Cálculo de orden de convergencia para varios N
N_list = [250, 500, 1000, 2000]
p_cn, dt_cn, err_cn = convergence_rate(F, U0, T, Crank_Nicolson, q=2, N_list=N_list)
print("Orden numérico aproximado de Crank_Nicolson:", p_cn)

# 4) Gráfico de la órbita
plt.figure()
plt.plot(U_CN[:, 0], U_CN[:, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Trayectoria orbital (Crank_Nicolson)")
plt.axis("equal")
plt.grid(True)
plt.show()

# 5) Gráfico log-log de error vs dt
plt.figure()
plt.loglog(dt_cn, err_cn, marker='o')
plt.xlabel("dt")
plt.ylabel("Error estimado en T")
plt.title("Convergencia Crank_Nicolson (pendiente ≈ %.2f)" % p_cn)
plt.grid(True, which='both')
plt.show()