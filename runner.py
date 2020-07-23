import argparse
import random
from .logger import Logger
import torch
import torch.nn as nn
import json
from types import FunctionType
import os.path as path

def run(dataset,model,parser_config=None,**kwargs):
    """
    

    Parameters
    ----------
    dataset : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    parser_config : TYPE, optional
        DESCRIPTION. The default is None.
    **kwargs : TYPE
        DESCRIPTION.

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
    parser.add_argument('--logdir',type=str,default='C:\\Users\\leo\\stage\\larva-vae-beta-vae\\experiments')
    parser.add_argument('--max_data',type=int,help="Mqax out number of data points for debug reasons")
    parser.add_argument('--weight_decay',type=float,default=0)
    parser.add_argument('--valid_cut',type=float,default=0.9)
    parser.add_argument('--no_log',action='store_true')
    parser.add_argument('--tqdm',action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--clip',type=float)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--data_device',type=str,choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--num_embeds',type=int,default=10000)
    parser.add_argument('--tb_projection',action='store_true')
    parser.add_argument('--no_config',action='store_true')
    base_args=['epochs']
    base_sec_args=['max_data','weight_decay']
    if parser_config: custom_config=parser_config(parser)
    opt=parser.parse_args()
    if parser_config and type(custom_config)==dict: 
        for k,v in custom_config.get('duplicate_dict',{}).items():
            opt.__dict__[v]=opt.__dict__[k]
    ##reload config
    if path.exists('config.json') and not opt.no_config:
        opt.__dict__.update(json.load('config.json'))

    torch.manual_seed(opt.seed)
    random.seed(opt.seed)
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    opt.device=device
    print(f"Device is {device} ")
    opt.verbose= not opt.no_verbose
    
    ##define Dataset and associated variables
    if callable(dataset): dataset=dataset(opt)
    loader_args=kwargs.get('loader_args',dict(coll_fn=None,num_worker=0,shuffle=True))
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
        sep='/' if '/' in opt.data else '\\'
        experiment_name=opt.data.split(sep)[-1].split('.')[0]+'_'  if \
            sep in opt.data else ""
        for key  in base_args: experiment_name+=f"{key}_{opt.__dict__[key]}_"
        for key  in base_sec_args: experiment_name+=f"{key}_{opt.__dict__[key]}_" if opt.__dict__[key] else ""
        
        if  type(parser_config)==str:
            experiment_name+=parser_config
        elif type(parser_config)==dict:
            for key  in parser_config['core_args']: experiment_name+=f"{key}_{opt.__dict__[key]}_"
            for key  in parser_config['secondary_args']: experiment_name+=f"{key}_{opt.__dict__[key]}_" if opt.__dict__[key] else ""
        experiment_name+=f"datadim_{opt.data_dim}"
        opt.experiment_name=experiment_name
    print(f"Running {experiment_name}")
    if device=='cuda': print(f"Data memory usage {torch.cuda.memory_allocated()}")
    
    ##create model

    if isinstance(model,FunctionType):
        model=model(opt)
    metrics_keys,loss_func=model.metrics_keys,model.loss_func
    model=model(opt).to(device=device)
    
        
    logger=Logger(experiment_name,metrics_keys,opt,model_str=str(model))
    optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr,
                               weight_decay=opt.weight_decay)
    
    for epoch in range(1,opt.epochs+1):
        t_start=time()
        model.train()
        iterable=tqdm(train_loader) if opt.tqdm else train_loader
        
        for x in iterable:
            x=x.to(device=device)
            optimizer.zero_grad()
            dic=loss_func(model,x,opt)
            dic['loss'].backward()
            if opt.clip: nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            model_log=model.metrics(opt)
            if model_log: logger.push_model_metrics_dict(model_log)
            
            optimizer.step()
            logger.push_train_metrics_dict(dic)
        logger.close_epoch(model,valid_loader,loss_func)
        if opt.verbose: print(f"Epoch {epoch} done in {time()-t_start}")
    logger.close(model,dataset)