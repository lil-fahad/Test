from __future__ import annotations

import json, os, time, random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

SEED=42
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def model_factory(name: str):
    if name=='resnet18':
        m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,10); return m
    if name=='mobilenet_v3_small':
        m=models.mobilenet_v3_small(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,10); return m
    if name=='efficientnet_b0':
        m=models.efficientnet_b0(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,10); return m
    raise ValueError(name)

def evaluate(model, loader, device):
    model.eval(); correct=total=0; loss_sum=0.0; ce=nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device, non_blocking=True),y.to(device, non_blocking=True)
            logits=model(x); loss=ce(logits,y)
            loss_sum += float(loss)*y.numel(); correct += int((logits.argmax(1)==y).sum()); total += y.numel()
    return {'loss': loss_sum/max(1,total), 'accuracy': correct/max(1,total)}

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('This production training job requires CUDA')
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision('high')
    out=Path('.novatrain/output'); out.mkdir(parents=True, exist_ok=True)
    data_root=Path('.novatrain/datasets')
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    test_tf=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    train_ds=datasets.CIFAR10(str(data_root),train=True,download=True,transform=train_tf)
    test_ds=datasets.CIFAR10(str(data_root),train=False,download=True,transform=test_tf)
    # Real but bounded first production cycle: 30k train / full 10k test.
    train_ds=Subset(train_ds, list(range(30000)))
    train_loader=DataLoader(train_ds,batch_size=256,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    suite={}
    for name in ['resnet18','mobilenet_v3_small','efficientnet_b0']:
        model=model_factory(name).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5)
        scaler=torch.amp.GradScaler('cuda')
        ce=nn.CrossEntropyLoss(label_smoothing=0.1)
        history=[]; best=-1.0; started=time.time()
        for epoch in range(1,6):
            model.train(); total=correct=0; loss_sum=0.0
            for x,y in train_loader:
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16):
                    logits=model(x); loss=ce(logits,y)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                n=y.numel(); total+=n; loss_sum+=float(loss)*n; correct+=int((logits.argmax(1)==y).sum())
            sched.step(); val=evaluate(model,test_loader,device)
            row={'epoch':epoch,'train_loss':loss_sum/total,'train_accuracy':correct/total,'val_loss':val['loss'],'val_accuracy':val['accuracy'],'lr':opt.param_groups[0]['lr']}
            history.append(row); print(name,row,flush=True)
            ckpt={'model':name,'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'metrics':row,'classes':train_ds.dataset.classes}
            torch.save(ckpt,out/f'{name}_last.pt')
            if val['accuracy']>best:
                best=val['accuracy']; torch.save(ckpt,out/f'{name}_best.pt')
        suite[name]={'best_accuracy':best,'history':history,'seconds':round(time.time()-started,2),'parameters':sum(p.numel() for p in model.parameters())}
        del model; torch.cuda.empty_cache()
    result={'ok':True,'request_id':os.environ.get('NOVATRAIN_REQUEST_ID'),'repository':os.environ.get('NOVATRAIN_REPOSITORY'),'commit':os.environ.get('NOVATRAIN_COMMIT_SHA'),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'suite':suite}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
