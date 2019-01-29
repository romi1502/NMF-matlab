import numpy as np
import scipy.sparse as ss



def sparse_NMF(V, R=3, n_iter=50, init_W=None, init_H=None, alternating_update=True, compute_loglikelihood=True):
    """
        (W, H, loglikelihood) = sparse_NMF(V, R=3, n_iter=50, init_W=None, init_H=None, alternating_update=True, compute_loglikelihood=True)
        NMF with Multinomial distribution of a sparse matrix. Columns of matrix V are modeled as multinomial draws of distribution P_n(f) = sum_z P(f|z) P_n(z). The log-likelihood L=sum_fn V_fn log(P_n(f)) is minimized with the EM algorithm.
        In the code W[f,z] = P(f|z) and H[z,n] = P_n(z). See Shashanka et al., Sparse Overcomplete Latent Variable Decomposition of Counts Data, NIPS 2007 for more details.
        The code is written for V to be a sparse matrix and could be applied to quite large sparse matrices (tested with 500kx500k matrices with density 0.001 with 30GB of RAM).
        Optimization might be done on the used sparse matrix type and on data redundancy (rows, cols, InvLambda, Lambda are redundant).

    Input :
       - V : matrix to be factorized (a MxN sparse matrix in format csr or coo).
       - R : number of templates
       - n_iter : number of iterations
       - init_W, init_H: initial values for W and H. Initialized with random uniform distribution if None.
       - alternating_update: whether to do one step update as in classical EM (no recomputation of P_n(f) between update of P(f|z) and P_n(z) or alternating update between P(f|z) and P_n(z) (with recomputation of P_n(f))
       - compute_loglikelihood: whether to compute the log-likelihood function.
    Output :
       - W : normalized frequency templates P(f|z) (MxR dense array)
       - H : temporal activation P_n(z) (RxN dense array)
       - loglikelihood : evolution of Log-likelihood L = sum_ft V_fn log(P_n(f)) where P_n(f) = sum_z P(f|z) P_n(z)

     Copyright (C) 2019 Romain Hennequin
    """
    supported_format = {"csr","coo"}
    assert V.format in supported_format, f"format of V is not supported: {V.format}. Should be in {supported_format}"

    eps = np.spacing(1).astype(V.dtype)

    # size of input matrix
    M = V.shape[0];
    N = V.shape[1];

    # initialization of H and W
    if init_H is not None:
        H = init_H
        R = init_H.shape[0]
    else:
        H = np.random.rand(R,N).astype(V.dtype)

    if init_H is not None:
        W = init_W;
        R = init_W.shape[1]
    else:
        W = np.random.rand(M,R).astype(V.dtype)

    # normalization of W and H
    W/= W.sum(axis=0, keepdims=True) + eps
    H/= H.sum(axis=0, keepdims=True) + eps

    # array to save the value of the log-likelihood
    loglikelihood = np.zeros(n_iter, dtype=V.dtype);

    # array for storing processed value of P_n(f)
    InvLambda = V.copy()

    # get non null entries positions of V
    rows,cols = V.nonzero()

    # iterative computation
    for iteration in range(n_iter):

        # computation of Lambda (= P_n(t))
        Lambda = sparse_product(W, H, rows, cols)

        if compute_loglikelihood:
            loglikelihood[iteration] = (V.data*np.log(Lambda.data)).mean()

        # update of W
        InvLambda.data = 1./Lambda.data
        W_updated = W * ((V.multiply(InvLambda)).dot(H.T) + eps)
        W_updated/= W_updated.sum(axis=0, keepdims=True) + eps
        if alternating_update:
            W = W_updated
            # recomputation of Lambda (= P_n(t))
            Lambda = sparse_product(W, H, rows, cols)
            InvLambda.data = 1./Lambda.data

        # update of H
        H*= (V.multiply(InvLambda).T.dot(W).T + eps)
        H/= H.sum(axis=0, keepdims=True) + eps

        if not alternating_update:
            W = W_updated

    return (W, H, loglikelihood)

def sparse_product(W, H, rows, cols):
    """
        compute the product of matrices W and H only at specified position in rows and cols.
        return a sparse matrix with the product.
    """

    val_vec_list = []

    # Do block processing to avoid memory explosion
    block_size = 10000
    for k in range(0,len(rows), block_size):
        val_vec_list.append(np.einsum('ij,ij->i', W[rows[k:k+block_size],:], H[:,cols[k:k+block_size]].T))

    val_vec = np.hstack(val_vec_list)

    return ss.coo_matrix((val_vec,(rows,cols)), shape=V.shape, dtype=W.dtype)


if __name__=="__main__":
    # simple test of sparse_NMF
    F = 20_000
    T = 20_000
    R = 10

    # initialize random sparse matrix with shape 20kx20k.
    V = ss.random(F,T,density=0.001, dtype="float32", format="coo")


    # compute NMF
    W,H,loglikelihood = sparse_NMF(V, R=R, n_iter=10, alternating_update=False)

    # check that likelihood is increasing at each iteration
    assert np.diff(loglikelihood).max()>=0

    # compute NMF with alternating updates
    W,H,loglikelihood = sparse_NMF(V, R=R, n_iter=10, alternating_update=True)

    # check that likelihood is increasing at each iteration
    assert np.diff(loglikelihood).max()>=0