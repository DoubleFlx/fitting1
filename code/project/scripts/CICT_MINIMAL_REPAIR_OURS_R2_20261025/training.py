"""Run the 100-epoch CICT schedule with epoch recovery."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import warnings
import read_guard

from common import ROOT, DATA, CONTRACT, AUDIT, OUT, PROTOCOL, now, read_json, sha, write_json, log, atomic_replace
from data_manifest import split_record, manifest_sha

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
sys.path.insert(0,str(ROOT/"project"))
sys.path.insert(0,str(ROOT/"project/scripts"))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import LetterBox
from ultralytics.data.build import seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import ModelEMA, init_seeds
from protocol_b_b2_model import FreezeBatchNormStats
from protocol_b_m1_model import M1AuditedV8DetectionLoss
from protocol_b_m1_common import bn_running_state_clone, state_max_abs_delta, tensor_state_hash, tensor_hash
from repair_pseudo import generate_high_confidence_pseudo
from cict_minimal_repair import ignore_empty_pseudo_images
from repair_runtime import verify_bindings, repair_identity, budget_check
from protocol_b_schedule import ScheduledSourceDataset, ProtocolBBatchSampler
from protocol_b_target_schedule import read_target_image_readonly

cv2.setNumThreads(1)
torch.set_num_threads(1)
RUNS=OUT/"RUNS"
ROLES=("PREFIX","SO_MATCHED","ST_REFERENCE","OURS","ORACLE")


def csv_rows(path):
    with Path(path).open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))


def rows_for(split):
    record=split_record(split,CONTRACT)
    p=Path(record["path"])
    assert sha(p)==record["sha256"]
    return csv_rows(p)


def config():
    return get_cfg(overrides={"task":"detect","mode":"train","model":"yolo11n.pt","data":str(DATA/"source_train.yaml"),
        "epochs":100,"imgsz":640,"batch":32,"workers":2,"seed":42,"deterministic":True,"single_cls":True,
        "optimizer":"SGD","lr0":.01,"lrf":.01,"momentum":.937,"weight_decay":.0005,"nbs":64,
        "warmup_epochs":3.,"warmup_bias_lr":.1,"warmup_momentum":.8,"box":7.5,"cls":.5,"dfl":1.5,
        "close_mosaic":10,"amp":True,"cache":False,"rect":False,"val":False,"plots":False,"patience":1000,
        "hsv_h":.015,"hsv_s":.7,"hsv_v":.4,"degrees":0.,"translate":.1,"scale":.5,"shear":0.,
        "perspective":0.,"flipud":0.,"fliplr":.5,"mosaic":1.,"mixup":0.,"cutmix":0.,"copy_paste":0.,
        "multi_scale":0.,"compile":False,"pretrained":True})


class CictModel(DetectionModel):
    def init_criterion(self):
        assert not getattr(self,"end2end",False)
        return M1AuditedV8DetectionLoss(self)

    def loss(self,batch,preds=None):
        sl,si=DetectionModel.loss(self,batch,preds)
        sp=int(self.criterion.m1_last_positive_assignments)
        before=bn_running_state_clone(self)
        target=batch.get("_cict_target")
        lam=float(batch.get("_cict_lambda",0.))
        oracle=bool(batch.get("_cict_oracle",False))
        count=int(target.get("_cict_count",0)) if target is not None else 0
        active=target is not None and lam>0 and (oracle or count>0)
        tl,ti=torch.zeros_like(sl),torch.zeros_like(si)
        tp=0
        target_keep_images=0;target_ignored_images=0
        if active:
            if oracle:
                with FreezeBatchNormStats(self):tl,ti=DetectionModel.loss(self,target)
                target_keep_images=int(target["img"].shape[0])
            else:
                with ignore_empty_pseudo_images(self.criterion,target["batch_idx"],int(target["img"].shape[0])) as keep:
                    with FreezeBatchNormStats(self):tl,ti=DetectionModel.loss(self,target)
                    target_keep_images=int(keep.sum().item())
                    target_ignored_images=int(keep.numel())-target_keep_images
            tp=int(self.criterion.m1_last_positive_assignments)
        elif target is not None and not oracle:
            target_ignored_images=int(target["img"].shape[0])
        after=bn_running_state_clone(self)
        delta,mismatch=state_max_abs_delta(before,after)
        assert delta==0 and not mismatch,"STOP_TARGET_BN_DRIFT"
        total=sl+lam*tl
        for x in (sl,si,tl,ti,total):
            if not torch.isfinite(x).all():raise FloatingPointError("STOP_NONFINITE_LOSS")
        weighted=float((lam*tl.detach().sum()).float().cpu())
        summed=float(total.detach().sum().float().cpu())
        self.last_diagnostics={"lambda":lam,"target_instances":count,"target_active":active,"oracle":oracle,
            "target_keep_images":target_keep_images,"target_ignored_images":target_ignored_images,
            "source_items":si.detach().float().cpu().tolist(),"target_items":ti.detach().float().cpu().tolist(),
            "source_raw_loss":float(sl.detach().sum().float().cpu()),"target_raw_loss":float(tl.detach().sum().float().cpu()),
            "weighted_target_loss":weighted,"weighted_loss_share":weighted/max(summed,np.finfo(np.float32).eps),
            "source_positive_anchors":sp,"target_positive_anchors":tp,"target_BN_max_delta":delta,
            "BN_after_source_sha256":tensor_state_hash(before),"BN_after_target_sha256":tensor_state_hash(after),
            "target_box_DFL_in_total":bool(active and count>0),"true_empty_target_loss_retained":bool(oracle and active and count==0)}
        return total,si+lam*ti


def new_model(device,seed=42):
    init_seeds(seed,deterministic=True)
    init=read_json(CONTRACT/"INITIALIZATION_FREEZE.json")["generic_COCO_weights"]
    assert sha(init["path"])==init["sha256"]
    pre=torch.load(init["path"],map_location="cpu",weights_only=False)["model"].float()
    model=CictModel(deepcopy(pre.yaml),ch=3,nc=1,verbose=False)
    model.load(pre,verbose=False)
    model.args=config();model.names={0:"animal"};model.nc=1
    for name,p in model.named_parameters():
        if ".dfl" in name:p.requires_grad_(False)
    return model.to(device)


class SourceViews(YOLODataset):
    """Keep the original buffer/FIFO immutable when returning a G view."""
    def __init__(self,*args,lookup=None,adapt_seed=42,**kwargs):
        self.lookup=lookup or {};self.adapt_seed=adapt_seed;self.absolute_epoch=1;self.components=[]
        super().__init__(*args,**kwargs)

    def __getitem__(self,index):
        self.components=[]
        result=super().__getitem__(index)
        result["_cict_components"]=json.dumps(self.components,separators=(",",":"))
        return result

    def load_image(self,i,*args,**kwargs):
        uid=Path(self.im_files[i]).stem
        assert uid in self.allowed_source_ids,"STOP_NON_SOURCE_TRAIN_READ"
        read_guard.check_image(self.im_files[i])
        original=super().load_image(i,*args,**kwargs)
        before=hashlib.sha256(original[0].tobytes()).hexdigest()
        flag=int.from_bytes(hashlib.sha256(f"M1_SS0_SOURCE_INTRADOMAIN_V1|ACTIVE|{self.adapt_seed}|{self.absolute_epoch}|{uid}".encode()).digest()[:8],"big")/float(1<<64)<.5
        rec=self.lookup.get(uid)
        active=bool(rec and flag and rec["status"]=="COMPLETE")
        result=original
        if active:
            read_guard.check_image(rec["cache_path"])
            bgr=cv2.imread(rec["cache_path"],cv2.IMREAD_COLOR)
            assert bgr is not None and bgr.shape[:2]==tuple(original[1])
            h,w=original[2]
            if bgr.shape[:2]!=(h,w):bgr=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_LINEAR)
            assert not np.shares_memory(bgr,original[0])
            result=(bgr,original[1],original[2])
        assert before==hashlib.sha256(original[0].tobytes()).hexdigest(),"STOP_SOURCE_BUFFER_MUTATION"
        self.components.append({"uid":uid,"active_coin":flag,"G_applied":active,
            "input_sha":before,"output_sha":hashlib.sha256(result[0].tobytes()).hexdigest()})
        return result


class Schedule:
    def __init__(self,seed):
        self.epochs=defaultdict(list)
        aa={r["key"]:r for r in self.read(DATA/f"SCHEDULES/s{seed}_source_augmentation.jsonl")}
        for row in self.read(DATA/f"SCHEDULES/s{seed}_source.jsonl"):
            self.epochs[row["epoch"]].append({**row,"sample_aug_seeds":aa[row["key"]]["sample_aug_seeds"]})
        self.targets=defaultdict(list)
        if seed!=42:
            for row in self.read(DATA/f"SCHEDULES/s{seed}_target.jsonl"):self.targets[row["epoch"]].append(row)

    @staticmethod
    def read(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
    def rows_for_epoch(self,epoch):return self.epochs[epoch]


def source_collate(samples):
    components=[s.pop("_cict_components") for s in samples]
    result=YOLODataset.collate_fn(samples)
    result["_cict_components"]=components
    return result


class TargetSampler(Sampler):
    def __init__(self,rows,epoch):self.rows=rows;self.epoch=epoch
    def __len__(self):return len(self.rows)
    def __iter__(self):
        for row in self.rows:yield [(uid,row["sample_view_seeds"][p],self.epoch,row["batch"],p) for p,uid in enumerate(row["uids"])]


class TargetViews(Dataset):
    def __init__(self,rows,lookup=None,active=None):
        self.rows={r["id"]:r for r in rows};self.lookup=lookup or {};self.active=active or {}
        self.lb=LetterBox(new_shape=(640,640),auto=False,scale_fill=False,scaleup=True,center=True,stride=32,
            padding_value=114,interpolation=cv2.INTER_LINEAR)
    def __len__(self):return len(self.rows)
    def __getitem__(self,token):
        uid,view_seed,epoch,batch,pos=token
        row=self.rows[uid]
        read_guard.check_image(row["image_path"])
        image,decoder=read_target_image_readonly(Path(row["image_path"]))
        params=self.lb.get_params({"img":image})
        teacher=self.lb.apply_image({"img":image},params)["img"]
        student=teacher
        rec=self.lookup.get(uid);apply=bool(rec and rec["status"]=="COMPLETE" and self.active.get((epoch,batch*8+pos),False))
        if apply:
            read_guard.check_image(rec["cache_path"])
            deployed=cv2.imread(rec["cache_path"],cv2.IMREAD_COLOR)
            assert deployed is not None and deployed.shape==image.shape
            student=self.lb.apply_image({"img":deployed},params)["img"]
        def tensor(img):return torch.from_numpy(np.ascontiguousarray(img[:,:,::-1].transpose(2,0,1)))
        return {"teacher_img":tensor(teacher),"img":tensor(student),"uid":uid,"view_seed":view_seed,
            "epoch":epoch,"batch":batch,"position":pos,"P_applied":apply,"orig_shape":params["orig_shape"],
            "ratio":params["ratio"],"pad":(params["left"],params["top"]),"decoder":decoder}


def target_collate(samples):
    return {k:torch.stack([s[k] for s in samples]) if k in ("img","teacher_img") else [s[k] for s in samples] for k in samples[0]}


def gt_target_batch(view,labels,device):
    boxes=[];indices=[]
    for i,uid in enumerate(view["uid"]):
        rec=labels[uid]
        assert rec["status"] in ("complete_boxes","confirmed_empty")
        h,w=view["orig_shape"][i];rx,ry=view["ratio"][i];left,top=view["pad"][i]
        for b in rec["boxes"]:
            cls,x,y,bw,bh=b
            assert cls==0
            boxes.append([(x*w*rx+left)/640,(y*h*ry+top)/640,bw*w*rx/640,bh*h*ry/640]);indices.append(i)
    return {"img":view["img"].to(device,non_blocking=True).float()/255,
        "cls":torch.zeros((len(boxes),1),device=device),"bboxes":torch.tensor(boxes,dtype=torch.float32,device=device).reshape(-1,4),
        "batch_idx":torch.tensor(indices,dtype=torch.long,device=device),"_cict_count":len(boxes)}


def pseudo_target_batch(view,ema,device):
    teacher=view["teacher_img"].to(device,non_blocking=True).float()/255
    ema.ema.eval()
    with torch.inference_mode(),torch.autocast("cuda",enabled=device.type=="cuda"):
        output=ema.ema(teacher)
    # Use local TorchNMS here.
    prior=sys.modules.pop("torchvision",None)
    try:
        pb=generate_high_confidence_pseudo(output,conf=.65,iou=.65,max_det=20,imgsz=640,area_min=.001,aspect_min=.2,aspect_max=5.)
    finally:
        if prior is not None:sys.modules["torchvision"]=prior
    return {"img":view["img"].to(device,non_blocking=True).float()/255,"cls":pb.cls,
        "bboxes":pb.bboxes,"batch_idx":pb.batch_idx,"_cict_count":len(pb.cls)},pb.stats


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state(),
        "torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def set_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:torch.cuda.set_rng_state_all(state["torch_cuda"])


def to_cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [to_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(to_cpu(v) for v in value)
    return deepcopy(value)


class Engine:
    def __init__(self,role,seed,device="cuda:0",gate=False):
        assert role in ROLES and (seed==42 if role=="PREFIX" else seed in (45,46))
        self.role=role;self.seed=seed;self.device=torch.device(device);self.gate=gate
        self.model=new_model(self.device)
        self.args=config()
        self.args.seed=seed
        self.model.args=self.args
        self.optimizer=DetectionTrainer.build_optimizer(SimpleNamespace(args=self.args,model=self.model),self.model,
            name="SGD",lr=.01,momentum=.937,decay=.0005,iterations=1)
        self.lf=lambda e:(1-e/100)*.99+.01
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=self.lf)
        self.scheduler.last_epoch=-1
        self.scaler=torch.amp.GradScaler("cuda",enabled=self.device.type=="cuda")
        self.ema=ModelEMA(self.model,decay=.9999,tau=2000)
        self.last_opt=-1;self.calls=0;self.unskipped=0;self.window_target=False;self.epoch=0
        self.source_gen=torch.Generator();self.target_gen=torch.Generator()
        self.source_gen.manual_seed(seed);self.target_gen.manual_seed(seed+1000)
        self.directory=RUNS/("OURS_REPAIR_s45" if role=="OURS" and seed==45 else f"GATE_ONLY_{role}_s{seed}")
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lookup={};self.active={};self.labels=None
        if role in ("OURS","ORACLE"):
            rec=read_json(DATA/f"METHOD_ASSETS/s{seed}/TRAINING_LOOKUP.json")
            self.lookup=rec
            import pandas as pd
            mask=pd.read_parquet(DATA/f"METHOD_ASSETS/s{seed}/P_NATIVE_ACTIVE_MASK.parquet")
            self.active={(int(r.epoch),int(r.target_visit_ordinal)):bool(r.valid) for r in mask.itertuples()}
        if role=="ORACLE":
            authorization=read_json(AUDIT/"ORACLE_GT_ACCESS_AUTHORIZATION.json")
            assert sha(authorization["GT_path"])==authorization["GT_sha256"]
            frozen=read_json(CONTRACT/"NON_ORACLE_END_POINTS_FREEZE.json")
            assert len(frozen["endpoints"])==6
            for rec in frozen["endpoints"].values():assert sha(rec["path"])==rec["sha256"]
            self.labels=read_json(authorization["GT_path"])
            assert set(self.labels)=={r["id"] for r in rows_for("target_adapt")}
        if DATA.name=="CICT_A_V2_SMALL_ORACLE":read_guard.install(role,seed)

    def step(self):
        self.scaler.unscale_(self.optimizer)
        found=getattr(self.scaler,"_per_optimizer_states",{}).get(id(self.optimizer),{}).get("found_inf_per_device",{})
        found_inf=any(float(v.detach().cpu())!=0 for v in found.values())
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.)
        if not torch.isfinite(norm) and not found_inf:raise FloatingPointError("STOP_NONFINITE_GRAD_NO_AMP_INF")
        called=False;original=self.optimizer.step
        def observed(*args,**kwargs):
            nonlocal called
            called=True;return original(*args,**kwargs)
        scale_before=self.scaler.get_scale();self.optimizer.step=observed
        try:self.scaler.step(self.optimizer)
        finally:self.optimizer.step=original
        self.scaler.update();self.optimizer.zero_grad();self.ema.update(self.model)
        self.calls+=1;self.unskipped+=int(called)
        row={"optimizer_calls":self.calls,"unskipped_updates":self.unskipped,"underlying_step_called":called,
            "AMP_found_inf":found_inf,"grad_norm":float(norm.detach().cpu()),"scale_before":scale_before,
            "scale_after":self.scaler.get_scale(),"EMA_updates":self.ema.updates,"target_contributing":bool(called and self.window_target)}
        self.window_target=False
        return row

    def payload(self,epoch):
        return {"schema":"CICT_V2_FULL_STATE_V1","protocol":PROTOCOL,"role":self.role,"seed":self.seed,"epoch":epoch,
            "repair_identity":repair_identity(),"experimental_arm":"OURS_REPAIR_s45",
            "horizon":100,"model":to_cpu(self.model.state_dict()),"ema":to_cpu(self.ema.ema.state_dict()),"ema_updates":self.ema.updates,
            "optimizer":to_cpu(self.optimizer.state_dict()),"scheduler":deepcopy(self.scheduler.state_dict()),"scaler":self.scaler.state_dict(),
            "rng":rng_state(),"source_generator":self.source_gen.get_state(),"target_generator":self.target_gen.get_state(),
            "pending_scaled_gradients":{n:to_cpu(p.grad) for n,p in self.model.named_parameters() if p.grad is not None},
            "last_opt_step":self.last_opt,"optimizer_calls":self.calls,"unskipped":self.unskipped,"window_target":self.window_target,
            "sampler_cursor":{"completed_epoch":epoch,"next_epoch":epoch+1,"next_source_batch":0,"next_target_batch":0,
                "worker_FIFOs":"discarded at epoch end; persistent_workers=False"},
            "dataset_freeze_sha256":sha(CONTRACT/"DATASET_FREEZE.json"),"schedule_freeze_sha256":sha(CONTRACT/"SCHEDULE_FREEZE.json"),
            "dataset_artifact_manifest_sha256":manifest_sha(CONTRACT),
            "ours_method_sha256":sha(CONTRACT/"OURS_METHOD_FREEZE.json"),"runner_sha256":sha(Path(__file__)),
            "architecture":deepcopy(self.model.yaml),"args":vars(self.args),"utc":now()}

    def save(self,epoch,registered=False,epoch_audit=None):
        path=self.directory/(f"e{epoch:03d}_full_state.pt" if registered else f"last_full_state_slot{epoch%2}.pt")
        if registered and path.exists():
            if path.with_suffix(".manifest.json").exists():raise RuntimeError("Registered state already exists; cannot overwrite")
            # Keep uncommitted files before rerunning the epoch.
            path.resolve().relative_to(RUNS.resolve())
            orphan=path.with_name(path.stem+"_UNCOMMITTED_"+sha(path)[:16]+path.suffix)
            assert not orphan.exists();path.rename(orphan)
        tmp=path.with_suffix(".pt.part")
        with tmp.open("wb") as f:
            torch.save(self.payload(epoch),f)
            f.flush();os.fsync(f.fileno())
        atomic_replace(tmp,path)
        rec={"path":str(path),"sha256":sha(path),"epoch":epoch,"role":self.role,"seed":self.seed,
            "utc":now(),"horizon":100,"optimizer_calls":self.calls,"EMA_updates":self.ema.updates,
            "raw_state_sha256":tensor_state_hash(self.model.state_dict()),"EMA_state_sha256":tensor_state_hash(self.ema.ema.state_dict()),
            "complete_full_state":True,"pending_gradients":sum(p.grad is not None for p in self.model.parameters())}
        rec.update(registered_epoch_checkpoint=registered,
            retention="immutable registered binary" if registered else "two generation rolling; historical hash is retained in epoch audit after slot reuse")
        if epoch_audit is not None:rec["epoch_audit"]=deepcopy(epoch_audit)
        write_json(path.with_suffix(".manifest.json"),rec)
        return rec

    def restore(self,rec,parent=False):
        assert sha(rec["path"])==rec["sha256"],"STOP_CHECKPOINT_HASH"
        state=torch.load(rec["path"],map_location="cpu",weights_only=False)
        if not parent:assert state["repair_identity"]==repair_identity(),"STOP_REPAIR_RESUME_IDENTITY_DRIFT"
        assert state["schema"]=="CICT_V2_FULL_STATE_V1" and state["protocol"]==PROTOCOL and state["horizon"]==100
        for file,key in (("DATASET_FREEZE.json","dataset_freeze_sha256"),("SCHEDULE_FREEZE.json","schedule_freeze_sha256"),("OURS_METHOD_FREEZE.json","ours_method_sha256")):
            assert state[key]==sha(CONTRACT/file),"STOP_STATE_IDENTITY"
        assert state.get("dataset_artifact_manifest_sha256")==manifest_sha(CONTRACT),"STOP_DATASET_ARTIFACT_MANIFEST_DRIFT"
        self.model.load_state_dict(state["model"],strict=True);self.ema.ema.load_state_dict(state["ema"],strict=True)
        self.ema.updates=state["ema_updates"];self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"]);self.scaler.load_state_dict(state["scaler"])
        for p in self.model.parameters():p.grad=None
        if parent:
            assert "repair_identity" not in state,"STOP_PARENT_IS_NOT_ORIGINAL_COMMON_E71"
            assert state["role"]=="PREFIX" and state["seed"]==42 and state["epoch"]==71
            # Reset gradients and the optimizer cursor.
            self.last_opt=-1;self.calls=0;self.unskipped=0;self.window_target=False
            init_seeds(self.seed,deterministic=True)
            self.source_gen.manual_seed(self.seed);self.target_gen.manual_seed(self.seed+1000)
        else:
            assert state["role"]==self.role and state["seed"]==self.seed
            assert state["runner_sha256"]==sha(Path(__file__)),"Runner changed during continuation"
            self.last_opt=state["last_opt_step"];self.calls=state["optimizer_calls"];self.unskipped=state["unskipped"]
            self.window_target=state["window_target"]
            parameters=dict(self.model.named_parameters())
            for n,g in state["pending_scaled_gradients"].items():parameters[n].grad=g.to(self.device)
            self.source_gen.set_state(state["source_generator"]);self.target_gen.set_state(state["target_generator"])
            set_rng(state["rng"])
        self.epoch=state["epoch"]
        assert tensor_state_hash(self.model.state_dict())==rec["raw_state_sha256"]
        assert tensor_state_hash(self.ema.ema.state_dict())==rec["EMA_state_sha256"]

    def loaders(self,schedule,epoch):
        os.environ["CICT_TRAIN_EPOCH"]=str(epoch)
        source_rows=rows_for("source_train")
        ds=SourceViews(img_path=str(DATA/"source_train/images"),imgsz=640,batch_size=32,augment=True,hyp=self.args,
            rect=False,cache=False,single_cls=True,stride=32,pad=0.,prefix="CICT Source: ",task="detect",
            data={"names":{0:"animal"},"nc":1,"channels":3},lookup=self.lookup.get("G"),adapt_seed=self.seed)
        ds.allowed_source_ids={r["id"] for r in source_rows};ds.absolute_epoch=epoch
        assert {Path(p).stem for p in ds.im_files}==ds.allowed_source_ids
        if not getattr(self,"source_label_semantics_verified",False):
            for label in ds.labels:
                uid=Path(label["im_file"]).stem
                lines=(DATA/"source_train/labels"/(uid+".txt")).read_text(encoding="utf-8").splitlines()
                expected=np.asarray([line.split() for line in lines if line.strip()],dtype=np.float32).reshape(-1,5)
                actual=np.concatenate((label["cls"],label["bboxes"]),axis=1)
                assert np.array_equal(actual,expected),"STOP_NATIVE_SOURCE_LABEL_CHANGE_OR_DEDUPLICATION"
            self.source_label_semantics_verified=True
        if epoch>=91:
            ds.mosaic=False
            ds.close_mosaic(deepcopy(self.args))
        wrapped=ScheduledSourceDataset(ds)
        sampler=ProtocolBBatchSampler(schedule,{Path(p).stem:i for i,p in enumerate(ds.im_files)},epoch)
        source=DataLoader(wrapped,batch_sampler=sampler,num_workers=2,pin_memory=self.device.type=="cuda",collate_fn=source_collate,
            worker_init_fn=read_guard.worker_init,generator=self.source_gen,persistent_workers=False)
        target=None
        if self.role in ("ST_REFERENCE","OURS","ORACLE"):
            td=TargetViews(rows_for("target_adapt"),self.lookup.get("P"),self.active)
            target=DataLoader(td,batch_sampler=TargetSampler(schedule.targets[epoch],epoch),num_workers=2,
                pin_memory=self.device.type=="cuda",collate_fn=target_collate,worker_init_fn=read_guard.worker_init,generator=self.target_gen,persistent_workers=False)
        return source,target

    def run_epoch(self,schedule,epoch,max_windows=None,no_step=False):
        start=time.perf_counter();self.model.train()
        # Freeze BN statistics only during target forwards.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",UserWarning);self.scheduler.step()
        source,target=self.loaders(schedule,epoch)
        target_iter=iter(target) if target is not None else None
        rows=[];updates=[];nb=len(source);nw=max(round(3*nb),100);before_calls=self.calls
        target_reads=0
        stream=None;update_stream=None;attempt=None
        if not self.gate:
            tag=now().replace(":","").replace(".","_")
            attempt=self.directory/"ATTEMPTS"/f"e{epoch:03d}_{tag}_p{os.getpid()}"
            attempt.mkdir(parents=True)
            stream=(attempt/"batches.jsonl").open("x",encoding="utf-8")
            update_stream=(attempt/"updates.jsonl").open("x",encoding="utf-8")
        previous_batch_end=time.perf_counter()
        source_iter=iter(source)
        for i,batch in enumerate(source_iter):
            batch_arrived=time.perf_counter()
            expected=schedule.rows_for_epoch(epoch)[i]
            assert list(batch["protocol_b_uid"])==expected["uids"],"STOP_SOURCE_SCHEDULE_DRIFT"
            assert list(batch["protocol_b_aug_seed"])==expected["sample_aug_seeds"],"STOP_SOURCE_AUG_SEED_DRIFT"
            ni=i+nb*(epoch-1)
            accumulate=max(1,int(np.interp(ni,[0,nw],[1,2]).round())) if ni<=nw else 2
            if ni<=nw:
                for group in self.optimizer.param_groups:
                    group["lr"]=np.interp(ni,[0,nw],[.1 if group.get("param_group")=="bias" else 0.,group["initial_lr"]*self.lf(epoch-1)])
                    if "momentum" in group:group["momentum"]=np.interp(ni,[0,nw],[.8,.937])
            for k in ("img","cls","bboxes","batch_idx"):batch[k]=batch[k].to(self.device,non_blocking=True)
            batch["img"]=batch["img"].float()/255
            target_stats={};view=None
            if target_iter is not None:
                view=next(target_iter);expected_t=schedule.targets[epoch][i]
                assert view["uid"]==expected_t["uids"] and view["view_seed"]==expected_t["sample_view_seeds"],"STOP_TARGET_SCHEDULE_DRIFT"
                target_reads+=len(view["uid"])
                if self.role=="ORACLE":tb=gt_target_batch(view,self.labels,self.device)
                else:tb,target_stats=pseudo_target_batch(view,self.ema,self.device)
                batch.update(_cict_target=tb,_cict_lambda=.05+.15*(epoch-72)/28,_cict_oracle=self.role=="ORACLE")
            with torch.autocast("cuda",enabled=self.device.type=="cuda"):
                loss,items=self.model(batch)
                loss=loss.sum()
            self.scaler.scale(loss).backward()
            diag=self.model.last_diagnostics
            self.window_target|=bool(diag["target_active"])
            did_step=False
            if ni-self.last_opt>=accumulate and not no_step:
                update=self.step();update.update(epoch=epoch,batch=i,global_microbatch=ni)
                if update_stream is not None:update_stream.write(json.dumps(update)+"\n")
                updates.append(update);self.last_opt=ni;did_step=True
            row={"epoch":epoch,"batch":i,"global_microbatch":ni,"source_N":len(expected["uids"]),
                "Source_UID_hash":hashlib.sha256("\n".join(expected["uids"]).encode()).hexdigest(),
                "Source_aug_seed_hash":hashlib.sha256(json.dumps(expected["sample_aug_seeds"]).encode()).hexdigest(),
                "Target_UID_hash":"" if view is None else hashlib.sha256("\n".join(view["uid"]).encode()).hexdigest(),
                "Target_view_seed_hash":"" if view is None else hashlib.sha256(json.dumps(view["view_seed"]).encode()).hexdigest(),
                "source_tensor_sha256":tensor_hash(batch["img"]) if i in (0,nb-1) or self.gate else "",
                "target_tensor_sha256":tensor_hash(batch["_cict_target"]["img"]) if view is not None and (i in (0,nb-1) or self.gate) else "",
                "target_visits":0 if view is None else len(view["uid"]),"P_applied":0 if view is None else sum(view["P_applied"]),
                "G_components":sum(len(json.loads(s)) for s in batch["_cict_components"]),
                "G_applied_components":sum(sum(c["G_applied"] for c in json.loads(s)) for s in batch["_cict_components"]),
                "source_components_sha256":hashlib.sha256("\n".join(batch["_cict_components"]).encode()).hexdigest(),
                "optimizer_step":did_step,"AMP_scale":self.scaler.get_scale(),"LR":[g["lr"] for g in self.optimizer.param_groups],
                **diag,**{"HPL_"+k:v for k,v in target_stats.items() if k not in ("confidences","areas")}}
            if self.gate and self.device.type=="cuda":torch.cuda.synchronize()
            batch_done=time.perf_counter()
            row.update(microbatch_wall_seconds=batch_done-previous_batch_end,
                loader_wait_seconds=batch_arrived-previous_batch_end,compute_wall_seconds=batch_done-batch_arrived)
            previous_batch_end=batch_done
            rows.append(row)
            if stream is not None:
                stream.write(json.dumps(row,ensure_ascii=False)+"\n")
                if (i+1)%16==0:
                    stream.flush();os.fsync(stream.fileno());update_stream.flush();os.fsync(update_stream.fileno())
            if no_step or (max_windows is not None and self.calls-before_calls>=max_windows):break
        # Stop workers before saving RNG state.
        for iterator in (source_iter,target_iter):
            if iterator is not None and hasattr(iterator,"_shutdown_workers"):iterator._shutdown_workers()
        del source_iter,target_iter,source,target
        if stream is not None:
            for f in (stream,update_stream):f.flush();os.fsync(f.fileno());f.close()
        read_guard.export(AUDIT/"READ_GUARDS"/f"{self.role}_s{self.seed}_e{epoch}_p{os.getpid()}_main.json")
        if self.device.type=="cuda":torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        if max_windows is None and not no_step:
            assert len(rows)==nb
            if self.role in ("ST_REFERENCE","OURS","ORACLE"):assert target_reads==nb*8
            from schedules import step_trace
            trace=step_trace(len(rows_for("source_train")),1 if self.role=="PREFIX" else 72,epoch)[-1]
            assert self.calls==trace["cumulative_phase_optimizer_calls"],"STOP_OPTIMIZER_SCHEDULE_DRIFT"
        self.ema.update_attr(self.model,include=["yaml","nc","args","names","stride","class_weights"])
        # Check parameters and EMA once per epoch.
        for value in self.model.state_dict().values():
            if value.is_floating_point() and not torch.isfinite(value).all():raise FloatingPointError("STOP_NONFINITE_STATE")
        self.epoch=epoch
        return rows,updates,{"epoch":epoch,"wall_seconds":elapsed,"source_batches":len(rows),"optimizer_calls":self.calls-before_calls,
            "EMA_updates":self.ema.updates,"target_training_reads":target_reads,"Dev_Test_training_reads":0,"dataset_mutation_count":0,
            "mean_source_loss":float(np.mean([r["source_raw_loss"] for r in rows])),
            "mean_weighted_target_loss":float(np.mean([r["weighted_target_loss"] for r in rows])),
            "target_active_batches":sum(r["target_active"] for r in rows),"HPL_total":sum(r.get("HPL_final_boxes",0) for r in rows),
            "first_source_tensor_sha256":rows[0]["source_tensor_sha256"],"last_source_tensor_sha256":rows[-1]["source_tensor_sha256"],
            "batch_file":None if attempt is None else str(attempt/"batches.jsonl"),
            "update_file":None if attempt is None else str(attempt/"updates.jsonl")}


def append_jsonl(path,rows):
    with Path(path).open("a",encoding="utf-8") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+"\n")
        f.flush();os.fsync(f.fileno())


def run(role,seed):
    process_start=time.perf_counter()
    assert role=="OURS" and seed==45,"STOP_ONLY_OURS_REPAIR_S45_AUTHORIZED"
    verify_bindings()
    for gate in ("G1_CPU.json","G2_IDENTITY.json","G3_REAL.json","G4_BUDGET.json"):
        assert read_json(AUDIT/gate)["status"]=="PASS",gate
    engine=Engine(role,seed)
    end=71 if role=="PREFIX" else 100
    latest=engine.directory/"LATEST_COMPLETE_STATE.json"
    # Finish a verified epoch commit without rerunning it.
    known=read_json(latest)["epoch"] if latest.exists() else (0 if role=="PREFIX" else 71)
    candidates=[]
    for manifest in engine.directory.glob("*.manifest.json"):
        rec=read_json(manifest)
        if rec["epoch"]>known and rec.get("epoch_audit") is not None:candidates.append(rec)
    for rec in sorted(candidates,key=lambda r:r["epoch"]):
        if rec["epoch"]!=known+1:break
        assert sha(rec["path"])==rec["sha256"],"STOP_UNCOMMITTED_FULL_STATE_CORRUPTION"
        audit=rec["epoch_audit"]
        assert sha(audit["batch_file"])==audit["batches_sha256"] and sha(audit["update_file"])==audit["updates_sha256"]
        ap=engine.directory/"EPOCH_AUDITS"/f"e{rec['epoch']:03d}.json"
        if not ap.exists():write_json(ap,{**audit,"checkpoint":rec})
        else:assert read_json(ap)["checkpoint"]["sha256"]==rec["sha256"]
        write_json(latest,rec);known=rec["epoch"]
        log("completed_epoch_commit_recovered_without_training",role=role,seed=seed,epoch=known,checkpoint_sha256=rec["sha256"])
    if latest.exists():engine.restore(read_json(latest))
    elif role!="PREFIX":engine.restore(read_json(CONTRACT/"CICT_SHARED_E71_PARENT.json"),parent=True)
    schedule=Schedule(seed)
    for epoch in range(engine.epoch+1,end+1):
        rows,updates,audit=engine.run_epoch(schedule,epoch)
        epochdir=engine.directory/"EPOCH_AUDITS";epochdir.mkdir(exist_ok=True)
        if (epochdir/f"e{epoch:03d}.json").exists():raise RuntimeError("Completed epoch evidence already exists")
        batchfile=Path(audit["batch_file"]);updatefile=Path(audit["update_file"])
        audit.update(role=role,seed=seed,batches_sha256=sha(batchfile),updates_sha256=sha(updatefile),
            teacher_student_distance_L2=math.sqrt(sum(float((a.detach().float()-b.detach().float()).square().sum().cpu())
                for a,b in zip(engine.model.parameters(),engine.ema.ema.parameters()))))
        checkpoint=engine.save(epoch,registered=epoch in ((71,) if role=="PREFIX" else (94,98,100)),epoch_audit=audit)
        write_json(epochdir/f"e{epoch:03d}.json",{**audit,"checkpoint":checkpoint})
        write_json(latest,checkpoint)  # Publish after the epoch commit.
        write_json(engine.directory/"RUN_STATE.json",{"utc":now(),"role":role,"seed":seed,"last_epoch":epoch,"complete":epoch==end,"checkpoint":checkpoint})
        print(json.dumps({"role":role,"seed":seed,"epoch":epoch,"seconds":audit["wall_seconds"],"complete":epoch==end}),flush=True)
        budget_check(time.perf_counter()-process_start,audit["wall_seconds"],100-epoch)
        assert not (OUT/"BUDGET_STOP_REQUEST.json").exists(),"STOP_REQUESTED_BUDGET_AT_COMPLETE_EPOCH"
        controller_stop=OUT/"CONTROLLER_OWN_TRAINING_STOP_REQUEST.json"
        if controller_stop.exists():
            assert role not in read_json(controller_stop)["roles"],"STOP_REQUESTED_BY_OWN_CONTROLLER_AT_COMPLETE_EPOCH"
        if role=="PREFIX" and epoch in (1,2):
            forecast=audit["wall_seconds"]*(71-epoch)+audit["wall_seconds"]*232*1.6+3600
            write_json(AUDIT/f"G4_GPU_FORECAST_PREFIX_E{epoch:03d}.json",{"utc":now(),"remaining_forecast_seconds":forecast,
                "cap_seconds":86400,"basis":"Actual Source epoch; conservative adaptation multiplier1.6 plus one-hour evaluation/gate reserve"})
            assert forecast<86400,"STOP_GPU_BUDGET_FORECAST"
    last=read_json(engine.directory/"LATEST_COMPLETE_STATE.json")
    if role=="PREFIX":write_json(CONTRACT/"CICT_SHARED_E71_PARENT.json",last)
    else:write_json(engine.directory/"ENDPOINT.json",last)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("role",choices=ROLES);parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args();run(args.role,args.seed)
