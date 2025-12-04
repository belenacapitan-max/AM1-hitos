from numpy import sqrt, array

def CR3BP(t, U, mu):
    x, y, z, vx, vy, vz = U

    r1 = sqrt((x + mu)**2 + y**2 + z**2)
    r2 = sqrt((x - 1 + mu)**2 + y**2 + z**2)

    #Derivadas del potencial efectivo
    dOmega_dx = -(1 - mu)*(x + mu)/r1**3 - mu*(x - 1 + mu)/r2**3 + x
    dOmega_dy = -(1 - mu)*y/r1**3 - mu * y/r2**3 + y
    dOmega_dz = -(1 - mu)*z/r1**3 - mu * z/r2**3

    #Aceleraciones
    ax = 2 * vy + dOmega_dx
    ay = - 2 * vx + dOmega_dy
    az = dOmega_dz

    return array([vx, vy, vz, ax, ay, az])

def make_CR3BP(mu):
    def f_cr3bp(t, U): # para que tenga la forma de f(t, U)
        return CR3BP(t, U, mu)
    return f_cr3bp