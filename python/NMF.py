import numpy as np

def NMF(V, R=3, n_iter=50, beta=0, init_W=[], init_H=[], verbose=False):
    """
     [W,H, cost] = NMF(V,R,n_iter,beta,initialV)
        NMF with beta divergence cost function.
    Input :
       - V : power spectrogram to factorize (a MxN matrix)
       - R : number of templates
       - n_iter : number of iterations
       - beta (optional): beta used for beta-divergence (default : beta = 0, IS divergence)
       - initialV (optional) : initial values of W, H (a struct with
       fields W and H)
    Output :
       - W : frequency templates (MxR array)
       - H : temporal activation
       - cost : evolution of beta divergence

     Copyright (C) 2015 Romain Hennequin
    """

    eta = 1;
    eps = np.spacing(1)

    # size of input spectrogram
    M = V.shape[0];
    N = V.shape[1];

    # initialization
    if len(init_H):
        H = init_H
        R = init_H.shape[0]
    else:
        H = np.random.rand(R,N);

    if len(init_H):
        W = init_W;
        R = init_W.shape[1]
    else:
        W = np.random.rand(M,R);

    # array to save the value of the beta-divergence
    cost = np.zeros(n_iter);

    # computation of Lambda (estimate of V) and of filters repsonse
    Lambda = np.dot(W,H);

    # iterative computation
    for it in range(n_iter):

        # compute beta divergence and plot its evolution
        cost[it] = beta_divergence(V+eps, Lambda+eps, beta);

        # update of W
        W*= (np.dot((Lambda**(beta-2.0)*V), H.T) + eps)/(np.dot(Lambda**(beta-1), H.T) + eps);

        # recomputation of Lambda (estimate of V)
        Lambda = np.dot(W,H) + eps;

        # update of H
        H*= (np.dot(W.T, Lambda**(beta-2)*V) + eps)/(np.dot(W.T, Lambda**(beta-1)) + eps);

        # recomputation of Lambda (estimate of V)
        Lambda = np.dot(W,H) + eps;

    return [W, H, cost]

def beta_divergence(V, Vh, beta):
    """
    Compute element-wise beta divergence between two matrices
    """
    if beta == 0:
        bD = (V/Vh-np.log(V/Vh) - 1).sum();
    elif beta == 1:
        bD = (V*(np.log(V)-np.log(Vh)) + Vh - V).sum();
    else:
        bD = max(1/(beta*(beta-1))*(V**beta + (beta-1)*Vh**beta - beta*V*Vh**(beta-1)),0).sum();

    return bD