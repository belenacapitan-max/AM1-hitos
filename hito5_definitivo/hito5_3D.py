import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # necesario para 3D
from matplotlib.animation import FuncAnimation

from cauchy_problem import Cauchy_problem
from temporal_schemes import RK4
from N_body_problem import make_F_Nbody, total_energy


# ---------- Parámetros del problema ----------
masses = np.array([1.0, 1.0, 0.1])   # tres masas
G = 1.0
dim = 3

# Posiciones iniciales en 3D 
r1 = np.array([-1.0, 0.0, 0.0])
r2 = np.array([ 1.0, 0.0, 0.0])
r3 = np.array([ 0.0, 1.5, 0.5])

# Velocidades iniciales en 3D 
v1 = np.array([ 0.0,  0.3,  0.4])
v2 = np.array([ 0.0, -0.3,  0.4])
v3 = np.array([ 0.0,  0.0, -0.8])

# Vector U0: por cada cuerpo [x,y,z,vx,vy,vz]
U0 = np.hstack((r1, v1, r2, v2, r3, v3))   # longitud 2*dim*N = 18

# Mallado temporal
T = 40.0
N = 4000      # pasos para integración
t = np.linspace(0.0, T, N+1)
dt = t[1] - t[0]

# F = dU/dt
F = make_F_Nbody(masses, G=G, dim=dim)

# Integración numérica
U = Cauchy_problem(F, U0, t, RK4)      # (N+1, 18)

# Reorganizamos: (tiempo, cuerpo, [pos(3), vel(3)])
U_mat = U.reshape(N+1, len(masses), 2*dim)
r = U_mat[:, :, :dim]    # posiciones: (N+1, 3, 3)

# Energía en el tiempo
E = np.array([total_energy(U[n, :], masses, G, dim=dim) for n in range(N+1)])

# ================== GRÁFICA ESTÁTICA DE TRAYECTORIAS ==================
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(r[:,0,0], r[:,0,1], r[:,0,2], label='Cuerpo 1')
ax.plot(r[:,1,0], r[:,1,1], r[:,1,2], label='Cuerpo 2')
ax.plot(r[:,2,0], r[:,2,1], r[:,2,2], label='Cuerpo 3')

ax.scatter(r[0,:,0], r[0,:,1], r[0,:,2], color='k', s=30, label='Posición inicial')

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Problema de 3 cuerpos en 3D")
ax.legend()
ax.grid(True)

# Ajustar límites para que quede un cubo bonito
all_x = r[:,:,0].ravel()
all_y = r[:,:,1].ravel()
all_z = r[:,:,2].ravel()
max_range = max(all_x.max()-all_x.min(),
                all_y.max()-all_y.min(),
                all_z.max()-all_z.min()) / 2.0
mid_x = (all_x.max()+all_x.min())/2.0
mid_y = (all_y.max()+all_y.min())/2.0
mid_z = (all_z.max()+all_z.min())/2.0
ax.set_xlim(mid_x-max_range, mid_x+max_range)
ax.set_ylim(mid_y-max_range, mid_y+max_range)
ax.set_zlim(mid_z-max_range, mid_z+max_range)

plt.tight_layout()
plt.show()

# ================== GRÁFICA DE ENERGÍA ==================
plt.figure(figsize=(7,5))
plt.plot(t, E)
plt.xlabel("t")
plt.ylabel("Energía total")
plt.title("Conservación de energía (RK4, dt=%.4f)" % dt)
plt.grid(True)
plt.tight_layout()
plt.show()

