import torch
from torch.utils.tensorboard import SummaryWriter
import os.path as path
import shutil
import numpy as np
import json


SEP='-'*40
class Logger:

    def __init__(self,experiment_name,metric_keys,opt,model_str=None):
        self.opt=opt
        if self.opt.no_log:
            return
        self.metric_keys=set(metric_keys)
        
        
        
        config_item={'model_str':model_str,
                     'exp_name':experiment_name}
        config_path=path.join(opt.logdir,'logger_config.json')
        if path.exists(config_path):
            with open(config_path) as fd:
                config_dict=json.load(fd)
            exp_id=int(list(config_dict.keys())[-1])+1
        else:
           exp_id=1 
           config_dict=dict()
        config_dict[exp_id]=config_item
        with open(config_path,'w+') as fd:
            json.dump(config_dict,fd,indent=2)
        
        self.tag=str(exp_id)
        self.summ_path=path.join(opt.logdir,self.tag)  
        self.sw=SummaryWriter(self.summ_path)
        
        print(f"Running {experiment_name} as exp_{exp_id}")
        if model_str:
            self.sw.add_text(f'model',model_str)
        self.epoch_log={k:[] for k in self.metric_keys}
        self.verbose=not opt.no_verbose
        self.t=0
        self.model_t=0
        self.valid_t=0
        self.epoch=0
        
    def save_fig(self,fig,fig_name):
        fig_path=path.join(self.summ_path,f'{fig_name}_{self.tag}')
        fig.savefig(fig_path)
        
    def push_train_metrics_dict(self,dic):
        if self.opt.no_log:
            return
        assert all([k in self.metric_keys for k in dic.keys()])
        
        for k,v in dic.items():
            if self.opt.debug and torch.isnan(v):
                raise Exception('nan')
            v=float(v)
            self.epoch_log[k].append(v)
            self.sw.add_scalar(f"{k}/train",v,self.t)
        self.t+=1
        
    def push_model_metrics_dict(self,dic):
        if 'scalar' in dic:
                for k,v in dic['scalar'].items():
                    self.sw.add_scalar(f"model/{k}",v,self.model_t)
        self.model_t+=1
                    
    def close_epoch(self,model,valid_loader=None):
        if self.opt.no_log:
            return

        device='cuda' if next(model.parameters()).is_cuda else 'cpu'
        if self.verbose:
            logstr=""
            for k,v in self.epoch_log.items():
                avg=sum(v)/len(v)
                logstr+=f"{k}: {avg}\n"
            print(f"Train Epoch {self.epoch+1} \n{logstr}{SEP}\n")
        self.epoch_log={k:[] for k in self.metric_keys}
        if valid_loader:
            valid_epoch_log={k:[] for k in self.metric_keys}
            with torch.no_grad():
                model.eval()

                for x in valid_loader:
                    x=x.to(device=device)
                    dic=model(x)
                    assert all([k in self.metric_keys for k in dic.keys()])
                    
                    for k,v in dic.items():
                        v=float(v)
                        valid_epoch_log[k].append(v)
                        self.sw.add_scalar(f"{k}/valid",v,self.valid_t)
                    self.valid_t+=1
                if self.verbose:
                    logstr=""
                    for k,v in valid_epoch_log.items():
                        avg=sum(v)/len(v)
                        logstr+=f"{k}: {avg}\n"
                    print(f"Valid Epoch {self.epoch+1} \n{logstr}{SEP}\n")

            
        self.epoch+=1
                    
                    

            
    def close(self,model,dataset):

        if self.opt.no_log:
            return
        torch.save({'state_dict':model.state_dict(),
                    'opt':self.opt}, path.join(self.summ_path,'mdl.pt'))

