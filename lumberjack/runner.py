import argparse
import random
from lumberjack.logger import Logger
import torch
import torch.nn as nn
import json
import os.path as path
from time import time
from tqdm import tqdm
import pdb
import traceback



def run(dataset,model,parser_config=None,logger_class=Logger,**kwargs):
    """
    

    Parameters
    ----------
    dataset : 
        Pytorch dataset, or function that takes opt and returns it
    model : nn.Module or function(namespace) ->  class(namespace) or class(namespace)
        Model to be trained. It can also be a function that takes opt and
        return a class that takes opt for initializing, or just the class
        
    logger_class : class inheritant from lumberjack.Logger
        we can use the standard logger class or one that inherits from it and
        implements a custom closing routine
        
    parser_config : function(argparse.ArgumentParser) -> dict or None, optional
        Adds additional arguments to standard parser and returns a dict of lists
        containing those arguments according to their importance, should you wish
        they be added to the experiment_name
        {'core_args': arguments that are always added to the name,
                
            'secondary_args': arguments that are added to the name if they are not NULL,

            'duplicate_dict':dict{k1:k2} hould you wish the argument k2 appear as
            k1 in the name
            }
    **kwargs : TYPE
        loader_args: arguments to be passed to dataloader.

    Raises
    ------
    Exception
        .

    Returns
    -------
    None.

    """
    parser=argparse.ArgumentParser()
    
    ##general DL args
    parser.add_argument('--exp_name',type =str)
    parser.add_argument('--no_verbose',action='store_true')
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--logdir',type=str,default='logs')
    parser.add_argument('--max_data',type=int,help="Mqax out number of data points for debug reasons")
    parser.add_argument('--weight_decay',type=float,default=0)
    parser.add_argument('--valid_cut',type=float,default=0.9)
    parser.add_argument('--no_log',action='store_true')
    parser.add_argument('--tqdm',action='store_true')
    parser.add_argument('--debug', type=str,default=None,choices=('anomaly','metric'))
    parser.add_argument('--clip',type=float)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--data_device',type=str,choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--num_embeds',type=int,default=10000)
    parser.add_argument('--tb_projection',action='store_true')
    parser.add_argument('--no_config',action='store_true')
    base_args=['lr','epochs']
    base_sec_args=['max_data','weight_decay']
    
    custom_config=None
    if parser_config: custom_config=parser_config(parser)
    opt=parser.parse_args()
    if parser_config and type(custom_config)==dict: 
        for k,v in custom_config.get('duplicate_dict',{}).items():
            opt.__dict__[v]=opt.__dict__[k]
    ##reload config
    if path.exists('config.json') and not opt.no_config:
        with open('config.json') as fd:
            opt.__dict__.update(json.load(fd))
    if 'opt_' in kwargs: opt.__dict__.update(kwargs['opt_'].__dict__ if not 
                             type(kwargs['opt_'])==dict else kwargs['opt_'])
    
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    opt.device=device
    print(f"Device is {device} ")
    opt.verbose= not opt.no_verbose
    
    ##define Dataset and associated variables
    if callable(dataset): dataset=dataset(opt)
    
    loader_args=kwargs.get('loader_args',dict(collate_fn=None,num_workers=0,shuffle=True))
    if callable(loader_args): loader_args=loader_args(opt)
     
    ##define loaders
    if opt.valid_cut==1.:
        train_loader=torch.utils.data.DataLoader(dataset,**loader_args)
        valid_loader=None
    else:
        ltds=int(opt.valid_cut*len(dataset))
        lvds=len(dataset)-ltds
        tds,vds=torch.utils.data.random_split(dataset,[ltds,lvds])
        train_loader=torch.utils.data.DataLoader(tds,**loader_args,batch_size=opt.batch_size)
        valid_loader=torch.utils.data.DataLoader(vds,**loader_args,batch_size=opt.batch_size)
    ##define experiment name
    if opt.exp_name:
        experiment_name=opt.exp_name
    else:
        sep=path.sep
        experiment_name=opt.data.split(sep)[-1].split('.')[0]+'_'  if hasattr(opt,'data') and sep in opt.data else ""
        for key  in base_args: experiment_name+=f"{key}_{opt.__dict__[key]}_"
        for key  in base_sec_args: experiment_name+=f"{key}_{opt.__dict__[key]}_" if opt.__dict__[key] else ""
        if  type(custom_config)==str:
            experiment_name+=custom_config
        elif type(custom_config)==dict:
            for key  in custom_config['core_args']: experiment_name+=f"{key}_{opt.__dict__[key]}_"
            for key  in custom_config['secondary_args']: experiment_name+=f"{key}_{opt.__dict__[key]}_" if opt.__dict__[key] else ""
        #experiment_name+=f"datadim_{opt.data_dim}"
        opt.experiment_name=experiment_name
    if device=='cuda': print(f"Data memory usage {torch.cuda.memory_allocated()}")
    
    ##create model


    model=model(opt)
    metrics_keys=model.metrics_keys
    model=model.to(device=device)
    print(metrics_keys)
        
    logger=logger_class(experiment_name,metrics_keys,opt,model_str=str(model))
    optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr,
                               weight_decay=opt.weight_decay)
    iterable=tqdm(train_loader) if opt.tqdm else train_loader
    
    for epoch in range(1,opt.epochs+1):
        model.train()
        t_start=time()
        for x in iterable:
            with torch.autograd.set_detect_anomaly(opt.debug=='anomaly'):
                x=x.to(device=device)
                optimizer.zero_grad()
                try:
                    dic=model(x)
                    dic['loss'].backward()
                    if opt.clip: nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                    if hasattr(model,'metrics'):
                        logger.push_model_metrics_dict(model.metrics(opt))
                except :
                    traceback.print_exc()
                    logger.close(model,dataset)
                    if opt.debug=='anomaly': pdb.set_trace()
            optimizer.step()
            logger.push_train_metrics_dict(dic)
        logger.close_epoch(model,valid_loader)
        if opt.verbose: print(f"Epoch {epoch} done in {time()-t_start}")
    logger.close(model,dataset)
    