from herramientas_matematicas import Newton

def Crank_Nicolson (U1, t1, t2, F):
    dt= t2 - t1
    a = U1 + dt/2 * F(U1) #Constante

    def G(x):
        return x -a - dt/2 * F(x)
    return Newton(G, U1)