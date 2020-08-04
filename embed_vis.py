import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations,product
from lumberjack.utils import choices
import seaborn as sns

def mult_distr_plot(data,labels ,cmap=None,**kwargs):
    
    figsize=kwargs.get('figsize',None)
    n_rows=kwargs.get('n_rows',3)
    n_cols=kwargs.get('n_cols',2)
    if not figsize: figsize=(5*n_cols,5*n_rows)
    
    all_labels=set(labels)
    max_lims,min_lims=np.max(data,axis=0),np.min(data,axis=0)
    fig,axs=plt.subplots(n_rows,n_cols,figsize=figsize)
    axs=axs.flatten()
    
    combs=list(combinations(range(data.shape[1]),2))
    if len(combs)>n_rows*n_cols:
        combs=choices(combs,n_rows*n_cols)
    for l,(ax,dims) in product(all_labels,zip(axs,combs)):
        filt=labels==l
        filtered=data[filt]
        sns.kdeplot(filtered[:,dims[0]],filtered[:,dims[1]],ax=ax
                   ,cmap=cmap[l] if cmap else None,n_levels=6, shade=False, 
                   shade_lowest=False,cbar=False)
        ax.set_xlabel(str(dims[0]))
        ax.set_ylabel(str(dims[1]))
        ax.set_xlim(min_lims[0],max_lims[0])
        ax.set_ylim(min_lims[1],max_lims[1])
    return fig