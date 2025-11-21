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
