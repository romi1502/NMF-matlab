import numpy as np
import sys
from scipy.ndimage.filters import convolve1d

def NMFD(V, R=3 ,T=10, n_iter=50, init_W=[], init_H=[]):
    """
     NMFD(V, R=3 ,T=10, n_iter=50, init_W=None, init_H=None)
        NMFD as proposed by Smaragdis (Non-negative Matrix Factor
        Deconvolution; Extraction of Multiple Sound Sources from Monophonic
        Inputs). KL divergence minimization. The proposed algorithm was
        corrected.
    Input :
       - V : magnitude spectrogram to factorize (is a FxN numpy array)
       - R : number of templates (unused if init_W or init_H is set)
       - T : template size (in number of frames in the spectrogram) (unused if init_W is set)
       - n_iter : number of iterations
       - init_W : initial value for W.
       - init_H : initial value for H.
    Output :
       - W : time/frequency template (FxRxT array, each template is TxF)
       - H : activities for each template (RxN array)

     Copyright (C) 2015 Romain Hennequin
    """

    """
     V : spectrogram FxN
     H : activation RxN
     Wt : spectral template FxR t = 0 to T-1
     W : FxRxT
    """
    eps = np.spacing(1)

    # data size
    F = V.shape[0];
    N = V.shape[1];

    # initialization
    if len(init_H):
        H = init_H
        R = H.shape[0]
    else:
        H = np.random.rand(R,N);

    if len(init_W):
        W = init_W
        R = W.shape[1]
        T = W.shape[2]
    else:
        W = np.random.rand(F,R,T);

    One = np.ones((F,N));
    Lambda = np.zeros((F,N));

    cost = np.zeros(n_iter)
    for it in range(n_iter):
        sys.stdout.write('Computing NMFD. iteration : %d/%d' % (it+1, n_iter));
        sys.stdout.write('\r')
        sys.stdout.flush()

        # computation of Lambda
        Lambda[:] = 0;
        for f in range(F):
            for z in range(R):
                cv = np.convolve(W[f,z,:],H[z,:]);
                Lambda[f,:] = Lambda[f,:] + cv[0:Lambda.shape[1]];

        Halt = H.copy();

        Htu = np.zeros((T,R,N));
        Htd = np.zeros((T,R,N));

        # update of H for each value of t (which will be averaged)
        VonLambda = V/(Lambda + eps);

        cost[it] = (V*log(V/Lambda)-V+Lambda).sum()

        Hu = np.zeros((R,N));
        Hd = np.zeros((R,N));
        for z in range(R):
            for f in range(F):
                cv = np.convolve(VonLambda[f,:],np.flipud(W[f,z,:]));
                Hu[z,:] = Hu[z,:] + cv[T-1:T+N-1];
                cv = np.convolve(One[f,:],np.flipud(W[f,z,:]));
                Hd[z,:] = Hd[z,:] + cv[T-1:T+N-1];

        # average along t
        H = H*Hu/Hd;

        # computation of Lambda

        Lambda[:] = 0;
        for f in range(F):
            for z in range(R):
                cv = np.convolve(W[f,z,:],H[z,:])
                Lambda[f,:] += cv[0:Lambda.shape[1]]

        mu = 0
        constraint = 1.02**np.arange(0,T)-1
        constraint[0:2] = 0

        SumTot = W.sum(axis=1)

        VonLambda = V/(Lambda + eps)


        # update of Wt
        for t in range(T):
            shift_H = np.roll(H,-t).T
            if t>0:
                shift_H[-t:,:]=0

            W[:,:,t] = W[:,:,t] * (  np.dot(VonLambda,shift_H) / (np.dot(One,shift_H) + eps + mu * constraint[t])  );

    return (W, H, cost)