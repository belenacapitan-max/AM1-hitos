import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm


# =================== Problema de Cauchy genérico ===================

def Cauchy_problem(F, U0, t, Temporal_scheme):
    """
    Resuelve U' = F(U) en la malla t con un esquema Temporal_scheme.
    """
    N = len(t) - 1
    Nv = len(U0)
    U = np.zeros((N + 1, Nv))
    U[0, :] = U0

    for n in range(N):
        U[n+1, :] = Temporal_scheme(U[n, :], t[n], t[n+1], F)

    return U


# =================== Oscilador lineal x'' + x = 0 ===================

def F(U):
    """
    U = [x, v], con v = x'.
    Devuelve dU/dt = [v, -x].
    """
    x, v = U[0], U[1]
    return np.array([v, -x])


def exact_oscillator(t, x0, v0):
    """
    Solución exacta de x'' + x = 0 con condiciones iniciales x(0)=x0, x'(0)=v0.
    Devuelve array de shape (len(t), 2) con [x_exact, v_exact].
    """
    x_exact = x0 * np.cos(t) + v0 * np.sin(t)
    v_exact = -x0 * np.sin(t) + v0 * np.cos(t)
    return np.vstack((x_exact, v_exact)).T


# =================== Esquemas temporales (1 paso) ===================

def Euler_explicito(U1, t1, t2, F):
    dt = t2 - t1
    return U1 + dt * F(U1)


def Jacobian(f, x):
    n = len(x)
    J = np.zeros((n, n))
    h = 1e-7
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = h
        J[:, j] = (f(x + dx) - f(x - dx)) / (2*h)
    return J


def Newton(f, x0, tol=1e-10, max_it=50):
    x = x0.copy()
    for i in range(max_it):
        Fx = f(x)
        if norm(Fx) < tol:
            break
        J = Jacobian(f, x)
        dx = np.linalg.solve(J, -Fx)
        x = x + dx
    return x


def Euler_inverso(U1, t1, t2, F):
    """
    Euler implícito (Inverse Euler):
        U_{n+1} = U_n + dt * F(U_{n+1})
    """
    dt = t2 - t1

    def G(x):
        return x - U1 - dt * F(x)

    return Newton(G, U1)


def Crank_Nicolson(U1, t1, t2, F):
    """
    Crank_Nicolson:
        U_{n+1} = U_n + dt/2 ( F(U_n) + F(U_{n+1}) )
    """
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


def leapfrog_oscillator(x0, v0, t):
    """
    Leap_Frog específico para x''+x=0.
    Integra solo x(t), y devuelve x, v aproximados.

    Esquema:
        v_{n+1/2} = v_{n-1/2} - dt * x_n
        x_{n+1}   = x_n + dt * v_{n+1/2}

    Aquí arrancamos con un paso de Euler para obtener v_{1/2}.
    """
    N = len(t) - 1
    dt = t[1] - t[0]

    x = np.zeros(N+1)
    v = np.zeros(N+1)

    x[0] = x0
    v[0] = v0

    # Paso inicial: estimamos v_{1/2} con medio paso de Euler
    v_half = v0 - 0.5*dt * x0

    for n in range(N):
        # x_n conocido, v_{n-1/2} almacenado en v_half
        # 1) v_{n+1/2}
        v_half = v_half - dt * x[n]
        # 2) x_{n+1}
        x[n+1] = x[n] + dt * v_half
        # velocidad aproximada en t_{n+1} (opcional)
        v[n+1] = v_half + 0.5*dt * x[n+1]

    # devolvemos en formato (N+1, 2): [x, v]
    return np.vstack((x, v)).T


# =================== Regiones de estabilidad ===================

def stability_function(method, z):
    """
    Función de estabilidad R(z) para métodos de 1 paso aplicados a y' = λ y, z = λ dt.

    Para Leap–Frog se trata aparte (no tiene R(z) simple).
    """
    if method == 'Euler':
        return 1 + z
    elif method == 'Euler_inv':
        return 1 / (1 - z)
    elif method == 'CN':
        return (1 + z/2) / (1 - z/2)
    elif method == 'RK4':
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    else:
        raise ValueError("Método no soportado en stability_function")


def stability_region_1step(method, x_min=-4, x_max=2, y_min=-3, y_max=3, n=400):
    """
     |R(z)| <= 1
    """
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y

    R = stability_function(method, Z)
    mask = np.abs(R) <= 1

    return X, Y, mask



def leapfrog_stability_polar(r_max=3, Nr=400, Ntheta=800):
    """
    z = r e^{i θ}.
    """
    r = np.linspace(0, r_max, Nr)
    theta = np.linspace(0, 2*np.pi, Ntheta)

    R, TH = np.meshgrid(r, theta)

    # z en forma compleja
    Z = R * np.exp(1j * TH)

    # raíces r1, r2 del método leapfrog
    disc = np.sqrt(1 + Z**2)
    r1 = Z + disc
    r2 = Z - disc

    # condición de estabilidad
    mask = np.maximum(np.abs(r1), np.abs(r2)) <= 1

    return R, TH, mask


# =================== MAIN: integración + gráficas ===================

if __name__ == "__main__":
    # ----- Parámetros del problema -----
    x0 = 1.0
    v0 = 0.0
    T = 20.0          # tiempo final
    N = 200           # número de pasos
    t = np.linspace(0, T, N+1)
    dt = t[1] - t[0]

    U0 = np.array([x0, v0])

    # ----- Solución exacta -----
    U_exact = exact_oscillator(t, x0, v0)

    # ----- Soluciones numéricas con métodos de 1 paso -----
    U_euler = Cauchy_problem(F, U0, t, Euler_explicito)
    U_ei    = Cauchy_problem(F, U0, t, Euler_inverso)
    U_cn    = Cauchy_problem(F, U0, t, Crank_Nicolson)
    U_rk4   = Cauchy_problem(F, U0, t, RK4)

    # ----- Leap–Frog específico -----
    U_lf = leapfrog_oscillator(x0, v0, t)

    # =================== 1) GRÁFICAS EN EL TIEMPO ===================
    plt.figure(figsize=(10, 6))
    plt.plot(t, U_exact[:, 0], 'k-',  label='Exacta x(t)')
    plt.plot(t, U_euler[:, 0], 'r--', label='Euler')
    plt.plot(t, U_ei[:, 0],    'b--', label='Euler inverso')
    plt.plot(t, U_cn[:, 0],    'g--', label='Crank–Nicolson')
    plt.plot(t, U_rk4[:, 0],   'm--', label='RK4')
    plt.plot(t, U_lf[:, 0],    'c-.', label='Leap–Frog')
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.title(f"Oscilador lineal x''+x=0, dt={dt:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =================== 2) ÓRBITAS EN EL ESPACIO DE FASE (x,v) ===================
    plt.figure(figsize=(6, 6))
    plt.plot(U_exact[:, 0], U_exact[:, 1], 'k-',  label='Exacta')
    plt.plot(U_euler[:, 0], U_euler[:, 1], 'r--', label='Euler')
    plt.plot(U_ei[:, 0],    U_ei[:, 1],    'b--', label='Euler inverso')
    plt.plot(U_cn[:, 0],    U_cn[:, 1],    'g--', label='Crank–Nicolson')
    plt.plot(U_rk4[:, 0],   U_rk4[:, 1],   'm--', label='RK4')
    plt.plot(U_lf[:, 0],    U_lf[:, 1],    'c-.', label='Leap–Frog')
    plt.xlabel("x")
    plt.ylabel("v")
    plt.title("Espacio de fase (x,v)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =================== 3) REGIONES DE ESTABILIDAD ===================
    # Métodos de 1 paso
    methods = [('Euler', 'Euler explícito'),
               ('Euler_inv', 'Euler inverso'),
               ('CN', 'Crank–Nicolson'),
               ('RK4', 'Runge–Kutta 4º')]

    plt.figure(figsize=(12, 10))
    for i, (short, name) in enumerate(methods, 1):
        X, Y, mask = stability_region_1step(short)
        plt.subplot(2, 2, i)
        plt.contourf(X, Y, mask, levels=[-0.5, 0.5, 1.5])
        plt.colorbar()
        plt.axhline(0, color='k', linewidth=0.5)
        plt.axvline(0, color='k', linewidth=0.5)
        # Línea donde vive el oscilador: z = i dt * (eigenvalues ~ ±i)
        # es decir, eje imaginario: x=0
        plt.axvline(0, color='r', linestyle='--', label='oscilador (Im eje)')
        plt.title(f"Región de estabilidad: {name}")
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Leap–Frog
    X, Y, mask = leapfrog_stability_polar()
    plt.figure(figsize=(6, 6))
    plt.contourf(X, Y, mask, levels=[-0.5, 0.5, 1.5])
    plt.colorbar()
    plt.axhline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='k', linewidth=0.5)
    plt.axvline(0, color='r', linestyle='--', label='oscilador (Im eje)')
    plt.title("Región de estabilidad: Leap–Frog")
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.legend()
    plt.tight_layout()
    plt.show()

