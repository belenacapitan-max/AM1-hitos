from numpy import linspace, array
from numpy import meshgrid, sqrt
import matplotlib.pyplot as plt

from three_body_problem import make_CR3BP
from Problema_Cauchy import Cauchy_problem
from temporal_schemes import RK_emb
from estabilidad import find_lagrange_points

# ================== PARÁMETROS Y CAMPOS ==================

# Parámetro de masa (por ejemplo Tierra–Luna)
mu = 0.0121505856

# Campo CR3BP con mu fijado: f(t, U)
f_cr3bp = make_CR3BP(mu)

# Puntos de Lagrange (L1..L5)
pts = find_lagrange_points(mu)


# ================== CONDICIONES INICIALES ==================

def initial_condition_around_L(name, eps_pos=1e-3, eps_vel=0.0):
    """
    Construye una condición inicial ligeramente perturbada
    alrededor del punto de Lagrange 'name' ('L1',..., 'L5').

    eps_pos: perturbación en posición (en y)
    eps_vel: perturbación en velocidad (vy)
    """
    x_star, y_star = pts[name]

    x0  = x_star
    y0  = y_star + eps_pos   # pequeña perturbación en y
    z0  = 0.0

    vx0 = 0.0
    vy0 = eps_vel            # opcional: perturbación en vy
    vz0 = 0.0

    return array([x0, y0, z0, vx0, vy0, vz0])


def simular_orbita_Lagrange(Lname, T_total=20.0, N=4000,
                            eps_pos=1e-3, eps_vel=0.0):
    """
    Integra una órbita alrededor del punto de Lagrange Lname
    usando tu esquema RK embebido y el CR3BP.

    Lname   : 'L1', 'L2', 'L3', 'L4', 'L5'
    T_total : tiempo final adimensional
    N       : número de pasos de salida (N+1 puntos)
    eps_pos : perturbación inicial en y
    eps_vel : perturbación inicial en vy
    """
    U0 = initial_condition_around_L(Lname, eps_pos=eps_pos, eps_vel=eps_vel)

    t0 = 0.0
    tf = T_total
    t  = linspace(t0, tf, N+1)

    U = Cauchy_problem(t, RK_emb, f_cr3bp, U0)

    return t, U


# ================== DIBUJO DE ÓRBITAS ALREDEDOR DE L1..L5 ==================

if __name__ == "__main__":

    plt.figure()

    # Masas principales
    x1 = -mu       # m1 en (-mu, 0)
    x2 = 1.0 - mu  # m2 en (1-mu, 0)
    plt.plot(x1, 0.0, "r*", ms=10, label="m1")
    plt.plot(x2, 0.0, "b*", ms=10, label="m2")

    # Lista de puntos de Lagrange
    L_names = ["L1", "L2", "L3", "L4", "L5"]

    for Lname in L_names:
        if Lname in ["L1", "L2"]:
            # Muy inestables 
            eps_pos = 1e-5
            T_total = 8.0
        elif Lname == "L3":
            # Inestable 
            eps_pos = 1e-3   
            T_total = 40.0   
        else:
            # L4, L5 (estables) 
            eps_pos = 1e-3
            T_total = 30.0

        t, U = simular_orbita_Lagrange(Lname, T_total=T_total,
                                       N=4000, eps_pos=eps_pos, eps_vel=0.0)

        x = U[0, :]
        y = U[1, :]

        plt.plot(x, y, label=f"órbita alrededor de {Lname}")

        # Posición exacta del punto de Lagrange
        x_star, y_star = pts[Lname]
        plt.plot(x_star, y_star, "ko")
        plt.text(x_star, y_star, f" {Lname}", fontsize=9)

    plt.xlabel("x (adimensional)")
    plt.ylabel("y (adimensional)")
    plt.title("Órbitas alrededor de los puntos de Lagrange en el CR3BP")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

