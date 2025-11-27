import numpy as np

def make_F_Nbody(masses, G=1.0, dim=3):
    """
    Devuelve F(U) para el problema N-cuerpos gravitatorio en 'dim' dimensiones.

    masses : array de longitud N
    dim    : dimensión espacial (2 o 3 normalmente)

    U tiene longitud 2*dim*N con estructura:
      [x1,...,xd1, vx1,...,vxd1,  x2,...,xd2, vx2,...,vxd2,  ...]
    es decir, por cuerpo: [pos(dim), vel(dim)].
    """
    masses = np.asarray(masses, dtype=float)
    N = len(masses)

    def F(U):
        U = np.asarray(U, dtype=float)
        U_mat = U.reshape(N, 2*dim)    # (N, 2*dim)
        r = U_mat[:, :dim]             # posiciones (N, dim)
        v = U_mat[:, dim:]             # velocidades (N, dim)

        a = np.zeros_like(r)           # aceleraciones (N, dim)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                rij = r[i] - r[j]                      # vector rij
                dist = np.linalg.norm(rij)
                if dist == 0.0:
                    continue
                a[i] += -G * masses[j] * rij / dist**3

        dUdt = np.hstack((v, a))       # (N, 2*dim): [v, a]
        return dUdt.ravel()            # (2*dim*N,)

    return F


def total_energy(U, masses, G=1.0, dim=3):
    """
    Energía total (cinética + potencial) en 'dim' dimensiones.
    """
    U = np.asarray(U, dtype=float)
    masses = np.asarray(masses, dtype=float)
    N = len(masses)

    U_mat = U.reshape(N, 2*dim)
    r = U_mat[:, :dim]     # (N, dim)
    v = U_mat[:, dim:]     # (N, dim)

    # Energía cinética
    v2 = np.sum(v**2, axis=1)          # |v_i|^2
    K = 0.5 * np.sum(masses * v2)

    # Energía potencial
    V = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rij = r[i] - r[j]
            dist = np.linalg.norm(rij)
            if dist == 0.0:
                continue
            V += -G * masses[i] * masses[j] / dist

    return K + V
