import numpy as np

def NMFD(V, R=3 ,T=10 ,n_iter=50, init_W=None, init_H=None):
    """
    % NMFD(V, R=3 ,T=10 ,n_iter=50, init_W=None, init_H=None)
    %    NMFD as proposed by Smaragdis (Non-negative Matrix Factor
    %    Deconvolution; Extraction of Multiple Sound Sources from Monophonic
    %    Inputs). KL divergence minimization. The proposed algorithm was
    %    corrected.
    %Input :   
    %   - V : magnitude spectrogram to factorize (is a MxN numpy array)
    %   - R : number of templates
    %   - T : template size (in number of frames in the spectrogram)
    %   - n_iter : number of iterations
    %   - init_W (optional) : initial value for W.
    %   - init_H (optional) : initial value for H.
    %Output :
    %   - W : time/frequency template (TxMxR array, each template is TxM)
    %   - H : activities for each template (RxN array)
    %
    % Copyright (C) 2015 Romain Hennequin
    """

    """
    % V : spectrogram MxN
    % H : activation RxN
    % Wt : spectral template MxR t = 0 to T-1
    % W : TxMxR
    """
    eps = np.spacing(1)

    # data size
    M = V.shape[0];
    N = V.shape[1];

    # initialization
    if init_H:
        H = init_H
    else:
        H = np.random.rand(R,N);

    if init_W:
        W = init_W
    else:
        W = np.random.rand(M,R,T);

    One = np.ones((M,N));
    Lambda = np.zeros((M,N));


    for iter in range(n_iter):
        
        # computation of Lambda
        Lambda[:] = 0;
        for f in range(M):
            for z in range(R):
                cv = np.convolve(W[f,z,:],H[z,:]);
                Lambda[f,:] = Lambda[f,:] + cv[0:Lambda.shape[1]];
            
        Halt = H.copy();
        
        Htu = np.zeros((T,R,N));
        Htd = np.zeros((T,R,N)); 
       
        # update of H for each value of t (which will be averaged)
        VonLambda = V/(Lambda + eps);
        
        Hu = np.zeros(R,N);    
        Hd = np.zeros(R,N);    
        for z in range(R):
            for f in range(M):
                cv = np.convolve(VonLambda[f,:],flipud(W[f,z,:]));
                Hu[z,:] = Hu[z,:] + cv[T-1:T+N-1];
                cv = np.convolve(One[f,:],flipud(W[f,z,:]));
                Hd[z,:] = Hd[z,:] + cv[T-1:T+N-1];
        
        
        # average along t
        H = H*Hu/Hd;

        # computation of Lambda

        Lambda[:] = 0;
        for f in range(M):
            for z in range(R):
                cv = conv(W[f,z,:],H[z,:])
                Lambda[f,:] += cv[0:Lambda.shape[1]]

        mu = 0
        constraint = 1.02**np.arange(0,T)-1
        constraint[0:2] = 0
        
        SumTot = W.sum(axis=1)
        
        VonLambda = V/(Lambda + eps)
        
        
        # update of Wt
        for t in range(T):
            W[:,:,t] = W[:,:,t] * (  (VonLambda*shiftLR(H,-t+1).T) / (One*shiftLR(H,-t+1).T + eps + mu * constraint[t])  );
        
        #print ['computing NMFD. iteration : ' int2str(iter) '/' int2str(n_iter)];
        
    return (W, H)