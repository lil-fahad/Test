from __future__ import annotations

import json, os, random, shutil, subprocess, time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

SEED = 42
EPOCHS = 5
MODELS = ['resnet18', 'mobilenet_v3_small', 'efficientnet_b0']
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

OUT = Path('.novatrain/output')
STATE = OUT / 'resume_state.json'
PROGRESS = OUT / 'progress.json'

def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def gpu_stats():
    try:
        s = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits'], text=True, timeout=3).strip().split(',')
        return {'gpu_util_pct': int(s[0]), 'vram_used_mb': int(s[1]), 'vram_total_mb': int(s[2]), 'gpu_temp_c': int(s[3])}
    except Exception:
        return {}

def disk_guard(min_free_gb=8.0):
    free = shutil.disk_usage(Path.cwd()).free / 1024**3
    if free < min_free_gb:
        raise RuntimeError(f'Low disk space: {free:.2f} GB free; safety floor is {min_free_gb:.2f} GB')
    return round(free, 2)

def model_factory(name):
    if name == 'resnet18':
        m=models.resnet18(weights=None); m.fc=nn.Linear(m.fc.in_features,10); return m
    if name == 'mobilenet_v3_small':
        m=models.mobilenet_v3_small(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,10); return m
    if name == 'efficientnet_b0':
        m=models.efficientnet_b0(weights=None); m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,10); return m
    raise ValueError(name)

def evaluate(model, loader, device):
    model.eval(); correct=total=0; loss_sum=0.0; ce=nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(x)
            loss=ce(logits.float(),y); n=y.numel(); total+=n
            loss_sum += float(loss)*n; correct += int((logits.argmax(1)==y).sum())
    return {'loss':loss_sum/max(total,1),'accuracy':correct/max(total,1)}

def save_progress(model_name, epoch, row, started, status='training'):
    elapsed=max(time.time()-started,0.001)
    done=(MODELS.index(model_name)*EPOCHS)+epoch
    total=len(MODELS)*EPOCHS
    p={'status':status,'model':model_name,'epoch':epoch,'epochs_per_model':EPOCHS,'completed_epochs':done,'total_epochs':total,'progress_pct':round(100*done/total,1),'elapsed_seconds':round(elapsed,1),'eta_seconds':round(elapsed/max(done,1)*(total-done),1),'metrics':row,'free_disk_gb':disk_guard(),**gpu_stats()}
    atomic_json(PROGRESS,p); print('NOVATRAIN_PROGRESS '+json.dumps(p),flush=True)

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda': raise RuntimeError('CUDA required')
    OUT.mkdir(parents=True,exist_ok=True); disk_guard()
    torch.backends.cudnn.benchmark=True; torch.set_float32_matmul_precision('high')
    train_tf=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
    test_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))])
    root=Path('.novatrain/datasets'); train_ds=datasets.CIFAR10(str(root),train=True,download=True,transform=train_tf); test_ds=datasets.CIFAR10(str(root),train=False,download=True,transform=test_tf)
    train_ds=Subset(train_ds,list(range(30000)))
    train_loader=DataLoader(train_ds,batch_size=256,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    test_loader=DataLoader(test_ds,batch_size=512,shuffle=False,num_workers=4,pin_memory=True,persistent_workers=True)
    suite={}; started=time.time()
    resume={}
    if STATE.exists():
        try: resume=json.loads(STATE.read_text(encoding='utf-8'))
        except Exception: resume={}
    for name in MODELS:
        disk_guard(); model=model_factory(name).to(device); opt=torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS); scaler=torch.amp.GradScaler('cuda'); ce=nn.CrossEntropyLoss(label_smoothing=.1)
        history=[]; best=-1.; start_epoch=1
        last=OUT/f'{name}_last.pt'
        if resume.get('model')==name and last.exists():
            ck=torch.load(last,map_location=device,weights_only=False); model.load_state_dict(ck['state_dict']); opt.load_state_dict(ck['optimizer']); start_epoch=int(ck['epoch'])+1; history=ck.get('history',[]); best=float(ck.get('best_accuracy',-1.));
            for _ in range(int(ck['epoch'])): sched.step()
            print(f'RESUMED {name} at epoch {start_epoch}',flush=True)
        for epoch in range(start_epoch,EPOCHS+1):
            model.train(); total=correct=0; loss_sum=0.
            for x,y in train_loader:
                x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.float16): logits=model(x); loss=ce(logits,y)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update(); n=y.numel(); total+=n; loss_sum+=float(loss)*n; correct+=int((logits.argmax(1)==y).sum())
            sched.step(); val=evaluate(model,test_loader,device); row={'epoch':epoch,'train_loss':loss_sum/total,'train_accuracy':correct/total,'val_loss':val['loss'],'val_accuracy':val['accuracy'],'lr':opt.param_groups[0]['lr']}; history.append(row)
            best=max(best,val['accuracy']); ck={'model':name,'epoch':epoch,'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'metrics':row,'history':history,'best_accuracy':best,'classes':train_ds.dataset.classes}; torch.save(ck,last)
            if val['accuracy']>=best: torch.save(ck,OUT/f'{name}_best.pt')
            atomic_json(STATE,{'model':name,'epoch':epoch,'updated_at':time.time()}); save_progress(name,epoch,row,started)
        suite[name]={'best_accuracy':best,'history':history,'parameters':sum(p.numel() for p in model.parameters())}; del model; torch.cuda.empty_cache()
        next_i=MODELS.index(name)+1
        if next_i < len(MODELS): atomic_json(STATE,{'model':MODELS[next_i],'epoch':0,'updated_at':time.time()})
    result={'ok':True,'request_id':os.environ.get('NOVATRAIN_REQUEST_ID'),'repository':os.environ.get('NOVATRAIN_REPOSITORY'),'commit':os.environ.get('NOVATRAIN_COMMIT_SHA'),'device':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'suite':suite}
    atomic_json(OUT/'metrics.json',result); atomic_json(PROGRESS,{'status':'succeeded','progress_pct':100.0,**gpu_stats()}); STATE.unlink(missing_ok=True); print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__': main()
