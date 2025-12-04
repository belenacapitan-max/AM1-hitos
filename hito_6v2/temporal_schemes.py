from numpy import zeros, array, dot, ceil
from numpy.linalg import norm
from herramientas_matematicas import Newton

def Euler_explicito(U1, t1, t2, F):
    dt = t2 - t1
    return U1 + dt * F(U1)

def Euler_implicito(U1, t1, t2, F):
    dt = t2 - t1
    
    def G(x):
        # x = U_{n+1}
        return x - U1 - dt * F(x)
    
    return Newton(G, U1)

def Crank_Nicolson(U1, t1, t2, F):
    dt = t2 - t1
    a = U1 + dt/2 * F(U1)

    def G(x):
        return x - a - dt/2 * F(x)

    return Newton(G, U1)

def RK4(U1, t1, t2, F):
    dt = t2 - t1
    k1 = F(U1)
    k2 = F(U1 + 0.5*dt*k1)
    k3 = F(U1 + 0.5*dt*k2)
    k4 = F(U1 + dt*k3)
    return U1 + dt*(k1 + 2*k2 + 2*k3 + k4)/6


def obtener_array_Butcher(q):


    if q==2: #RK orden 2-1
        N = 2
        a = zeros((N,N))
        b = zeros((N))
        bs = zeros((N))
        c = zeros((N))

        a[1, :] = [1, 0]

        b[:] = [1/2, 1/2]

        c[:] = [0, 1]

        bs[:] = [1, 0] #Orden 1 por ser embebido
    elif q==3: #RK orden 3-2
        N = 4
        a = zeros((N,N))
        b = zeros((N))
        bs = zeros((N))
        c = zeros((N))

        a[1, :] = [1/2, 0, 0, 0]
        a[2, :] = [0, 3/4, 0, 0]
        a[3, :] = [2/9, 1/3, 4/9, 0]

        b[:] = [2/9, 1/3, 4/9, 0]
        bs[:] = [7/24, 1/4, 1/3, 1/8]

        c[:] = [0, 1/2, 3/4, 1]

    elif q == 5:
        N = 6
        a = zeros((N,N))
        b = zeros((N))
        bs = zeros((N))
        c = zeros((N))

        # c
        c[:] = [
            0.0,
            1.0/4.0,
            3.0/8.0,
            12.0/13.0,
            1.0,
            1.0/2.0
        ]

        # a
        a[1, 0] = 1.0/4.0

        a[2, 0] = 3.0/32.0
        a[2, 1] = 9.0/32.0

        a[3, 0] = 1932.0/2197.0
        a[3, 1] = -7200.0/2197.0
        a[3, 2] = 7296.0/2197.0

        a[4, 0] = 439.0/216.0
        a[4, 1] = -8.0
        a[4, 2] = 3680.0/513.0
        a[4, 3] = -845.0/4104.0

        a[5, 0] = -8.0/27.0
        a[5, 1] = 2.0
        a[5, 2] = -3544.0/2565.0
        a[5, 3] = 1859.0/4104.0
        a[5, 4] = -11.0/40.0

        # b: orden 5 (alto)
        b[:] = [
            16.0/135.0,
            0.0,
            6656.0/12825.0,
            28561.0/56430.0,
            -9.0/50.0,
            2.0/55.0
        ]

        # bs: orden 4 (bajo)
        bs[:] = [
            25.0/216.0,
            0.0,
            1408.0/2565.0,
            2197.0/4104.0,
            -1.0/5.0,
            0.0
        ]

    else:
        raise ValueError(f"Orden q={q} no implementado en obtener_array_Butcher")
    return a, b, bs, c

def k_calculation(f, U, t, h, a, c):
    
    """
    Calcula las etapas k_i de un método RK (embebido o no).

    f : función f(t, U)
    U : estado actual (vector 1D: len(U))
    t : tiempo actual
    h : tamaño de paso
    a : matriz (s,s) coeficientes de Butcher
    c : vector (s,) nodos

    Devuelve:
        k : matriz (s, dimU) con las etapas k_i
    """
    U = array(U, dtype=float)
    s = len(c)
    dim = len(U)
    k = zeros((s, dim))

    for i in range(len(c)):
        #combinación lineal de etapas anteriores
        Ui = U + h * dot(a[i, :], k)
        ti = t + c[i] * h
        k[i, :] = f(ti, Ui) # pendientes intermedias

    return k

def RK_embedded_step(f, U, t, h, a, b, bs, c):
    """
    Un paso de Runge-Kutta embebido.

    Devuelve:
        U_high : solución de orden alto (b)
        U_low  : solución de orden bajo (bs)
        err_est: estimación de error (vector)
    """
    k = k_calculation(f, U, t, h, a, c)

    U_high = U + h * dot(b, k)
    U_low = U + h * dot(bs, k)

    error= U_high - U_low # Error de la estimación

    return U_high, U_low, norm(error)

def Embedded_RK(U0, t0, tf, f, q, tol ):

    dt = tf - t0

    a, b, bs, c = obtener_array_Butcher(q)

    # Estimación del error
    Uh, Ul, error = RK_embedded_step(f, U0, t0, dt, a, b, bs, c)
    # print(error)
    if error == 0:
        return Uh # Evita dividir por cero
    
    a, b, bs, c = obtener_array_Butcher(q)

    # Tamaño de subpaso 
    h = dt * (tol/ error)**(1.0/q)
    if h > dt:
        h = dt

    N = int(ceil(dt/ h))
    h = dt/ N

    U = array(U0, dtype = float)
    t = t0

    for _ in range(N):
        U, _, _ = RK_embedded_step(f, U, t, h, a, b, bs, c)
        t += h

    return U

def RK_emb(U0, t0, tf, f):
    q = 3
    tol = 1e-6
    return Embedded_RK(U0, t0, tf, f, q, tol)
