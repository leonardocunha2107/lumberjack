import torch
from torch.utils.tensorboard import SummaryWriter
import os.path as path
import shutil
import numpy as np

SEP='-'*40
class Logger:

    def __init__(self,experiment_name,metric_keys,opt,model_str=None):
        self.opt=opt
        if self.opt.no_log:
            return
        self.metric_keys=set(metric_keys)
        self.summ_path=path.join(opt.logdir,experiment_name)
        if path.isdir(self.summ_path):
            if not opt.no_verbose: print(f"Deleting logs at {self.summ_path}")
            shutil.rmtree(self.summ_path, ignore_errors=True)
        self.sw=SummaryWriter(self.summ_path)
        self.tag=experiment_name
        if model_str:
            self.sw.add_text(f'model/{self.tag}',model_str)
        self.epoch_log={k:[] for k in self.metric_keys}
        self.verbose=not opt.no_verbose
        self.t=0
        self.model_t=0
        self.valid_t=0
        self.epoch=0
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
                    
    def close_epoch(self,model,valid_loader=None,valid_metrics_fn=None):
        if self.opt.no_log:
            return
        assert valid_loader==None or valid_metrics_fn!=None
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
                    dic=valid_metrics_fn(model,x,self.opt)
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
        embeds,labels=dataset.get_embeds(model,self.opt)
        if opt.tb_projection:
            self.sw.add_embedding(embeds,tag=f'embed',
                                      metadata=labels)
            
        self.sw.close()
