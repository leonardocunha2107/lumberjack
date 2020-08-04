import torch
from os import path
import numpy as np
from itertools import product

def rand_indices(n,k):
    return torch.randperm(n).tolist()[:k]
def choices(inp,k):
    return [inp[idx] for idx in rand_indices(len(inp),k)]
def model_from_path(path,device=None):
    if not device:
        device='cuda' if torch.cuda.is_available() else 'cpu'
    if root_log_path not in path: path=path.join(root_log_path,path)
    if 'model_' not in path or 'model_vae' in path:
        from vae import VAE as mdl_class
    elif 'model_vrnn' in path:
        from vrnn import VRNN as mdl_class
    else: raise NotImplemented
    aux=torch.load(path,map_location=torch.device(device))
    state_dict,opt=aux['state_dict'],aux['opt']
    model=mdl_class(opt).to(device=device)
    model.load_state_dict(state_dict)
    return model,opt
def sample_from_cell(mins,maxs,num_samples=100,device=None):
    if not device:
        device='cuda' if torch.cuda.is_available() else 'cpu'
    min_tensor=torch.tensor(mins,device=device,dtype=torch.float)
    max_tensor=torch.tensor(maxs,device=device,dtype=torch.float)
    ts=torch.rand(num_samples,*min_tensor.shape,dtype=torch.float).to(device=device)
    ts=ts*(max_tensor-min_tensor).unsqueeze(0)+min_tensor.unsqueeze(0)
    return ts

def get_grid(mins,maxs,ks,grid_type='lin'):
    if grid_type=='norm':
        from scipy.stats import norm
        distr=norm()
        grid=[]
        for m,M,k in zip(mins,maxs,ks):
            bounds_for_range = distribution.cdf([m, M])
            pp = np.linspace(*bounds_for_range, num=k)
            grid.append(distribution.ppf(pp))
    else:
        grid=[np.linspace(m,M,k) for m,M,k in zip(mins,maxs,ks)]
    return np.array(grid)
def grid_generator(grid,ks,return_idxs=False):
    from itertools import product
    drange=list(range(len(ks)))
    for idxs in product(*(range(k-1) for k in ks)):
        mins,maxs=grid[drange,idxs], grid[drange,[idx+1 for idx in idxs]]
        if return_idxs: yield mins,maxs,idxs
        else: yield mins,maxs
def eval_grid(func,mins=[-1.5,-1.5],maxs=None,ks=[6,6],grid_type='lin',
              return_grid=False,**sample_args):
    assert len(ks)==len(mins)
    if not maxs: maxs=[-m for m in mins]
    grid=get_grid(mins,maxs,ks,grid_type)
    stats=np.empty(tuple(k-1 for k in ks),dtype=object)
    for mins,maxs,idxs in iter(grid_generator(grid,ks,return_idxs=True)):
        inp=sample_from_cell(mins,maxs,**sample_args)
        x=func(inp)
        stats[idxs]={'stat':x,'grid':(mins,maxs)}
    if return_grid: return stats,grid
    return stats