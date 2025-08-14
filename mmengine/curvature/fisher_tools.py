
import torch
from tqdm import tqdm

def kf_eigens(fisher: dict)-> dict:
    r"""
    Calculate layer-wise eigen-spectrum of the KFAC factors
    """
    n = len(fisher.keys())
    eigvals = dict()
    eigvecs = dict()

    progress_bar = tqdm(enumerate(fisher.items()))
    for (i, (name, (xxt, ggt))) in progress_bar:
        # print(f"Computing eigenvalues/vectors for layer {name} index :[{i}/{n}]")
        try:
            sym_xxt, sym_ggt = xxt + xxt.t(), ggt + ggt.t()
            #regularize
            sym_xxt += 1e-10*torch.eye(sym_xxt.shape[0]).to(sym_xxt.device)
            sym_ggt += 1e-10*torch.eye(sym_ggt.shape[0]).to(sym_ggt.device)

            xxt_eigvals, xxt_eigvecs = torch.linalg.eigh(sym_xxt)
            ggt_eigvals, ggt_eigvecs = torch.linalg.eigh(sym_ggt)

            eigvecs[name]=(xxt_eigvecs, ggt_eigvecs)
            eigvals[name]=(xxt_eigvals, ggt_eigvals)

        except Exception as err:
            print("\n")
            print(f'Error in layer {name} index :[{i}/{n}]')
            print(err)
            eigvals[name] = None
            eigvecs[name] = None


    return eigvals, eigvecs


def invert_cholesky(fisher: dict,
                    eigvals: dict(),
                    )-> dict:
    """Compute inverse cholesky of each fisher (Q,H) component 
    Input: dict: fisher[name] = (Q,H)
    Output: dict: invchol[name] = (ch(Q)^{-1}, ch(H)^{-1}"""


    invchol = dict()

    n = len(fisher.keys())

    #regularize so that (Q,H) are symmetric positive-definite and non-singular
    eps = 1e-10

    progress_bar = tqdm(enumerate(fisher.items()))

    for index, (name, value) in progress_bar:

        if eigvals[name]:     
            q_eigs, h_eigs = eigvals[name]
            min_q_eigs, min_h_eigs = torch.min(q_eigs), torch.min(h_eigs)
            first_eps = torch.abs(min_q_eigs) if min_q_eigs < 0. else 0.
            second_eps = torch.abs(min_h_eigs) if min_h_eigs < 0. else 0.
        else:
            first_eps, second_eps = 0. , 0. 

        first, second = value

        first = first + (first_eps + eps)*torch.eye(first.shape[0]).to(first.device)
        second = second + (second_eps + eps)*torch.eye(second.shape[0]).to(second.device)

        first = (first + first.t()) / 2.0
        second = (second + second.t()) / 2.0

        try:
            inv_chol_frst = torch.pinverse(torch.linalg.cholesky(first))
            inv_chol_scnd = torch.pinverse(torch.linalg.cholesky(second))

            invchol[name] = (inv_chol_frst, inv_chol_scnd)

        except Exception as err:
            print("\n")
            print(f"Error in layer {name} index :[{index}/{n}]")
            print(err)
            invchol[name] = (torch.zeros(first.shape, device = first.device), torch.zeros(second.shape, device = second.device))

    return invchol


def invert_fisher(
                  invchol: dict
                  )-> dict:
    """Compute inverse of each fisher (Q,H) component
     from the inverse cholesky 
    use formula: 
    Q^{-1} = ch(Q)^{-1} @ (ch(Q)^{-1}).t
    H^{-1} = ch(H)^{-1} @ (ch(H)^{-1}).t
    """

    invfisher = dict()

    n = len(invchol.keys())
    print(f"Computing inverse fisher ...")
    for index, (name, value) in tqdm(enumerate(invchol.items())):
        first, second = value

        try:
            inv_frst = first@first.t()
            inv_scnd = second@second.t()

            invfisher[name] = (inv_frst, inv_scnd)

        except Exception as err:
            print("\n")
            print(f"Error in layer {name} index :[{index}/{n}]")
            print(err)
            invfisher[name] = (torch.zeros(first.shape, device = first.device), torch.zeros(second.shape, device = second.device))

    return invfisher



def kf_inner(grad_1: dict,
             grad_2: dict,
             inverse_fisher: dict) -> float:
    r'''
    inner product of gradients in KF form, w.r.t. inverse fisher metric

    The inner product <x,y>_A, w.r.t a symmetric bilinear form A, is the matrix product x^T@A@y

    For every layer `name` compute: grad_1[name].t @ inverse_fisher[name] @ grad_2[name]
    Use the kronecker factorization
    grad_1[name] = (q_1, h_1)
    grad_2[name] = (q_2, h_2)
    inverse_fisher[name] = (Q^{-1}, H^{-1})

    (q_1\otimes h_1)^T @ (Q^{-1}, H^{-1}) @ (q_2\otimes h_2)
    = (q_1^T @ Q^{-1} @ q_2) \otimes (h_1^T @ H^{-1} @ h_2)
    '''

    assert len(grad_1) == len(grad_2) == len(inverse_fisher), "gradient and inverse fisher dimension mismatch"

    n = len(inverse_fisher)

    layer_inner_products = torch.empty(len(grad_1))

    for idx , (key, value) in enumerate(inverse_fisher.items()):

        forward_1, backward_1 = grad_1[key]
        forward_2, backward_2 = grad_2[key]
        Q_inv, H_inv = value

        try:
            q = forward_1.t() @ Q_inv @ forward_2
            h = backward_1.t() @ H_inv @ backward_2

            layer_inner_products[idx] = q*h

        except Exception as error:
            print(f"Error in layer [{idx}]/[{n}]")
            print(error)

    return torch.sum(layer_inner_products)



def random_fisher(size, device):
    """Generate a random positive definite symmetric matrix"""
    # Generate random normal matrix
    A = torch.randn(size, size, device=device)
    
    # Make symmetric (A + A^T)/2
    symmetric = (A + A.T) / 2
    
    # Ensure positive definite by adding a multiple of identity
    # The smallest eigenvalue will be at least 'min_eig'
    min_eig = 0.1  # You can adjust this as needed
    return symmetric + (min_eig + torch.abs(torch.rand(1, device=device))) * torch.eye(size, device=device)

