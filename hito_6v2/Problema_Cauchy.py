from numpy import zeros, array

def Cauchy_problem(t, temporal_scheme, f, U0):
    U = array (zeros((len(U0),len(t))))
    U[:,0] = U0
    for ii in range(0, len(t) - 1):
        U[:,ii+1] = temporal_scheme(U[:,ii], t[ii], t[ii+1], f)
    return U