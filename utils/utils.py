import torch

def torch_kron(A, B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

def torch_block_diag(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)
    shapes = torch.tensor([a.shape for a in arrs])
    i = []
    v = []
    r, c = 0, 0
    for k, (rr, cc) in enumerate(shapes):
        first_index = torch.arange(r, r + rr, device=arrs[0].device)
        second_index = torch.arange(c, c + cc, device=arrs[0].device)
        index = torch.stack((first_index.tile((cc,1)).transpose(0,1).flatten(), second_index.repeat(rr)), dim=0)
        i += [index]
        v += [arrs[k].flatten()]
        r += rr
        c += cc
    out_shape = torch.sum(shapes, dim=0).tolist()

    if arrs[0].device == "cpu":
        out = torch.sparse.DoubleTensor(torch.cat(i, dim=1), torch.cat(v), out_shape)
    else:
        out = torch.cuda.sparse.DoubleTensor(torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), out_shape)
    return out
    
def gradient(y, x, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs = grad_outputs, create_graph=True, retain_graph=True, allow_unused=True)[0]
    return grad

def computeFisherSum(grad):
    fisher = torch.ger(grad,grad)
    return fisher.abs().sum()