"""Fit CICT Native P and Source G views from SourceTrain and Adapt."""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from common import ROOT, DATA, CONTRACT, AUDIT, OUT, cpu_mode, now, read_json, sha, write_json, log
from data_manifest import split_record, manifest_sha
from resource_budget import cpu_cap_seconds, budget_identity, forecast_path, scaling_path, retained_measurements

POLICY = DATA / "METHOD_ASSETS"
Q2 = ROOT / "project/audits/M1_BLIND_PHOTOMETRIC_STYLE_SUPPORT_AUDIT_V1_20260808_120520/CODE_SNAPSHOT"
NS = "M1_SS0_SOURCE_INTRADOMAIN_V1"


def check_controller_stop(label):
    path=OUT/"CONTROLLER_OWN_TRAINING_STOP_REQUEST.json"
    if path.exists():
        assert label not in read_json(path).get("stage_labels",[]),"STOP_REQUESTED_BY_OWN_CONTROLLER_AFTER_COMPLETE_ASSET_BATCH"


def imports():
    sys.path.insert(0, str(ROOT / "project"))
    sys.path.insert(0, str(ROOT / "project/scripts"))
    import numpy as np
    import pandas as pd
    import m1_routea_photometric_realization_core_v1 as core
    import m1_routea_semantic_resource_policy_v1 as route
    core.configure_q2_snapshot(Q2)
    return np, pd, core, route


def h(role, *parts):
    return hashlib.sha256("|".join((NS, role, *map(str, parts))).encode()).hexdigest()


def image_rows():
    import csv
    rows = []
    for split in ("source_train", "target_adapt"):
        record = split_record(split, CONTRACT)
        p = Path(record["path"])
        assert sha(p) == record["sha256"]
        with p.open(encoding="utf-8", newline="") as f:
            rows.extend({**r, "domain": split, "canonical_uid": r["id"],
                "location_id": r["location"], "capture_event_id": r["event_key"]} for r in csv.DictReader(f))
    assert len({r["id"] for r in rows}) == len(rows)
    return rows


def descriptor_worker(row):
    cpu_mode()
    np, _, core, _ = imports()
    assert sha(row["image_path"]) == row["image_sha256"]
    rgb, decoder = core.decode_q2_rgb(row["image_path"])
    raw = core.q2_exact_raw_descriptor(rgb)
    assert np.isfinite(raw).all()
    return {"canonical_uid": row["id"], "domain": row["domain"], "location_id": row["location"],
        "capture_event_id": row["event_key"], "image_path": row["image_path"], "image_sha256": row["image_sha256"],
        "height": rgb.shape[0], "width": rgb.shape[1], "decoder": decoder,
        **{"whole__"+name: float(raw[i]) for i, name in enumerate(core.BASE)}}


def reuse_frozen_policy_assets():
    path = CONTRACT / "POLICY_SELECTION_FREEZE.json"
    if not path.exists():
        return False
    frozen = read_json(path)
    assert frozen["dataset_freeze_sha256"] == sha(CONTRACT / "DATASET_FREEZE.json")
    assert frozen["dataset_artifact_manifest_sha256"] == manifest_sha(CONTRACT)
    assert frozen["ours_method_sha256"] == sha(CONTRACT / "OURS_METHOD_FREEZE.json")
    for rec in frozen["files"]:
        assert sha(rec["path"]) == rec["sha256"], "STOP_FROZEN_POLICY_ASSET_MUTATION"
    log("frozen_policy_assets_verified_and_reused", files=len(frozen["files"]),
        policy_freeze_sha256=sha(path), policy_files_rewritten=0)
    return True


def descriptors():
    if reuse_frozen_policy_assets():
        return
    _, pd, _, _ = imports()
    POLICY.mkdir(parents=True, exist_ok=True)
    rows = image_rows()
    done = POLICY / "DESCRIPTORS.json"
    values = read_json(done) if done.exists() else {}
    assert set(values).issubset({r["id"] for r in rows})
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = [r for r in rows if r["id"] not in values]
        for offset in range(0, len(pending), 32):
            for rec in pool.map(descriptor_worker, pending[offset:offset+32]):
                values[rec["canonical_uid"]] = rec
            write_json(done, values)
            write_json(OUT / "METHOD_PREPARATION_STATE.json", {"utc": now(), "stage": "DESCRIPTORS",
                "complete": len(values) == len(rows), "processed": len(values), "total": len(rows),
                "session_wall_seconds": time.perf_counter()-start, "workers": 2, "GT_reads": 0, "Dev_Test_reads": 0})
            check_controller_stop("METHOD_DESCRIPTORS")
    pd.DataFrame([values[r["id"]] for r in rows]).to_csv(POLICY / "DESCRIPTOR_ROWS.csv", index=False)
    log("method_descriptors_complete", wall_seconds=time.perf_counter()-start, rows=len(values), GT_reads=0)


def geometry(frame, median, scale, label):
    np, pd, core, route = imports()
    from scipy.spatial import cKDTree
    from m1_routea_descriptor_geometry_measurement_v1 import weighted_quantile, unweighted_quantile
    events = []
    for ev, group in frame.groupby("capture_event_id", sort=True):
        events.append({"capture_event_id": ev, "domain": group.domain.iloc[0],
            **{name: unweighted_quantile(group["whole__"+name].to_numpy(), .5) for name in core.BASE}})
    events = pd.DataFrame(events)
    raw = events[core.BASE].to_numpy(dtype=np.float64)
    z = (raw-median)/scale
    zi = (frame[["whole__"+name for name in core.BASE]].to_numpy(dtype=np.float64)-median)/scale
    assert len(z) > 1 and np.isfinite(z).all() and np.isfinite(zi).all()
    n = len(z)
    nn = np.empty(n)
    edges_a, edges_b, distances = [], [], []
    for start in range(0, n, 32):
        stop = min(n, start+32)
        d = np.sum(np.abs(z[start:stop,None,:]-z[None,:,:])*core.WEIGHTS, axis=2, dtype=np.float64)
        d[np.arange(stop-start), np.arange(start,stop)] = np.inf
        nn[start:stop] = d.min(axis=1)
    tau = unweighted_quantile(nn, .95)
    degrees = np.empty(n, dtype=np.int64)
    for start in range(0, n, 32):
        stop = min(n, start+32)
        d = np.sum(np.abs(z[start:stop,None,:]-z[None,:,:])*core.WEIGHTS, axis=2, dtype=np.float64)
        d[np.arange(stop-start), np.arange(start,stop)] = np.inf
        a, b = np.nonzero(d <= 2*tau)
        edges_a.append(a+start); edges_b.append(b); distances.append(d[a,b])
        degrees[start:stop] = (d <= 2*tau).sum(axis=1)
    aa, bb, dd = map(np.concatenate, (edges_a, edges_b, distances))
    assert len(dd) and (degrees > 0).any(), "No empirical support graph; no substituted method"
    w = 1/(int((degrees>0).sum())*degrees[aa].astype(np.float64))
    q10, q90 = weighted_quantile(dd,w,.1), weighted_quantile(dd,w,.9)
    keep = (dd>=q10)&(dd<=q90)
    values, weights = dd[keep], w[keep]
    weights /= weights.sum(dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cdf = np.cumsum(weights, dtype=np.float64); cdf[-1] = 1.
    tree = cKDTree(z*core.WEIGHTS)
    tube, _ = tree.query(zi*core.WEIGHTS, k=1, p=1, eps=0, workers=1)
    internal, reason = route.validity_arrays(zi, median, scale)
    event_ids = events.capture_event_id.to_numpy(dtype=str)
    uids = frame.canonical_uid.to_numpy(dtype=str)
    np.savez(POLICY / (label+"_GEOMETRY.npz"), median=median, scale=scale, event_std=z, image_std=zi,
        event_ids=event_ids, image_uids=uids, image_event_ids=frame.capture_event_id.to_numpy(dtype=str),
        edge_a=aa, edge_b=bb, edge_distance=dd, dose_distance=values, dose_weights=weights,
        dose_cdf=cdf, tau=np.asarray(tau), image_tube_distance=tube)
    neighbors = {str(ev): (event_ids[bb[aa==i]], dd[aa==i]) for i, ev in enumerate(event_ids)}
    return SimpleNamespace(median=median, scale=scale, tau=tau, pmf_values=values, pmf_weights=weights, pmf_cdf=cdf,
        event_ids=event_ids, event_index={v:i for i,v in enumerate(event_ids)}, z_event=z, event_tree=tree,
        neighbors=neighbors, image_uids=uids, image_index={v:i for i,v in enumerate(uids)},
        image_event_ids=frame.capture_event_id.to_numpy(dtype=str), z_image=zi,
        original_in_tube=tube<=tau, original_internal_valid=internal, original_invalid_reason=np.asarray(reason),
        schedule_to_q2_uid={u:u for u in uids})


def source_panel(frame):
    queues = {}
    for loc, group in frame.groupby("location_id", sort=True):
        events = {ev: deque(sorted(eg.canonical_uid, key=lambda u:h("PANEL_UID",u)))
            for ev,eg in group.groupby("capture_event_id",sort=True)}
        queues[str(loc)] = deque((ev,events[ev]) for ev in sorted(events,key=lambda ev:h("PANEL_EVENT",loc,ev)))
    selected = []
    locations = sorted(queues,key=lambda loc:h("PANEL_LOCATION",loc))
    while len(selected)<512:
        for loc in locations:
            if queues[loc]:
                ev,uids = queues[loc].popleft(); selected.append(uids.popleft())
                if uids: queues[loc].append((ev,uids))
                if len(selected)==512: break
    return selected


def source_jobs(frame, ctx, seed, schedule):
    np, _, core, route = imports()
    visits = defaultdict(list)
    for r in schedule:
        for pos,uid in enumerate(r["uids"]): visits[uid].append((r["epoch"],r["batch"]*32+pos))
    jobs, candidates = [], []
    for row in frame.to_dict("records"):
        uid = row["canonical_uid"]; ii=ctx.image_index[uid]; ev=row["capture_event_id"]; a=ctx.event_index[ev]
        refs, dist = ctx.neighbors[ev]; valid=[]
        for epoch, ordinal in visits[uid]:
            if route.stateless_u01(NS+"|CANDIDATE_ACTIVE",seed,epoch,ordinal,uid,"AUGMENT_ON") >= .5: continue
            c = {"uid":uid,"epoch":epoch,"source_ordinal":ordinal,"valid":False,"reason":""}
            if not ctx.original_in_tube[ii]:
                c["reason"]="UNAVAILABLE_ORIGINAL_IMAGE_OFF_TUBE"; candidates.append(c); continue
            u=route.stateless_u01(NS+"|DOSE",seed,epoch,ordinal,uid,"INTENDED_GEOMETRIC_DOSE")
            ds=float(ctx.pmf_values[min(int(np.searchsorted(ctx.pmf_cdf,u,side="left")),len(ctx.pmf_values)-1)])
            legal=np.flatnonzero(dist>=ds)
            if not len(legal) or not ds>0:
                c["reason"]="NATIVE_TSVA_UNAVAILABLE_NO_DESTINATION_AT_DOSE"; candidates.append(c); continue
            pick=int(legal[route.stateless_index(len(legal),NS+"|DESTINATION",seed,epoch,ordinal,uid,"DESTINATION")])
            b=ctx.event_index[str(refs[pick])]
            target_event=ctx.z_event[a]+(ds/dist[pick])*(ctx.z_event[b]-ctx.z_event[a])
            delta=target_event-ctx.z_event[a]; target=ctx.z_image[ii]+delta
            legal_endpoint,_=route.validity_arrays(np.stack([target_event,target]),ctx.median,ctx.scale)
            d,_=ctx.event_tree.query(np.stack([target_event,target])*core.WEIGHTS,k=1,p=1,eps=0,workers=1)
            c.update(dose=ds,destination_event=str(refs[pick]),delta=delta.tolist(),target_std=target.tolist(),delta_sha256=route.delta_hash(delta))
            c["valid"]=bool(legal_endpoint.all() and (d<=ctx.tau).all())
            if not c["valid"]: c["reason"]="ENDPOINT_INTERNAL_OR_TUBE_UNAVAILABLE"
            candidates.append(c)
            if c["valid"]: valid.append(c)
        selected=valid[int(h("FIXED_SELECTOR",uid),16)%len(valid)] if valid else None
        jobs.append(make_job(row,ctx,seed,"S",selected,len(valid)))
    return jobs,candidates


def make_job(row,ctx,seed,label,selected,count):
    job={"uid":row["canonical_uid"],"seed":seed,"view":label,"image_path":row["image_path"],
        "image_sha256":row["image_sha256"],"event_id":row["capture_event_id"],
        "original_std":ctx.z_image[ctx.image_index[row["canonical_uid"]]].tolist(),
        "selected_candidate":selected,"valid_candidate_count":count,
        "geometry_file":str(POLICY/(label+"_GEOMETRY.npz"))}
    job["task_key"]=hashlib.sha256(json.dumps(job,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return job


def freeze_policies():
    if reuse_frozen_policy_assets():
        return
    np,pd,core,_=imports()
    from m1_routea_descriptor_geometry_measurement_v1 import weighted_quantile,unweighted_quantile
    from m1_fixed_fresh_seed_v2_prepare import _native_epoch
    import m1_routea_semantic_resource_policy_v1 as route
    from m1_native_a_fixed_policy_realization_v1 import selector_digest
    frame=pd.read_csv(POLICY/"DESCRIPTOR_ROWS.csv",dtype={"location_id":str,"capture_event_id":str})
    sources=frame[frame.domain=="source_train"].reset_index(drop=True)
    targets=frame[frame.domain=="target_adapt"].reset_index(drop=True)
    ev=[]
    for (dom,event),group in frame.groupby(["domain","capture_event_id"],sort=True):
        ev.append((dom,[unweighted_quantile(group["whole__"+n].to_numpy(),.5) for n in core.BASE]))
    raw=np.asarray([v for _,v in ev],dtype=np.float64)
    domains=[d for d,_ in ev]
    weights=np.asarray([.5/domains.count(d) for d in domains])
    med=np.asarray([weighted_quantile(raw[:,j],weights,.5) for j in range(7)])
    scale=np.asarray([weighted_quantile(raw[:,j],weights,.75)-weighted_quantile(raw[:,j],weights,.25)+1e-6 for j in range(7)])
    pctx=geometry(targets,med,scale,"P")
    sr=raw[np.asarray(domains)=="source_train"]
    smed=np.asarray([unweighted_quantile(sr[:,j],.5) for j in range(7)])
    sscale=np.asarray([unweighted_quantile(sr[:,j],.75)-unweighted_quantile(sr[:,j],.25)+1e-6 for j in range(7)])
    sctx=geometry(sources,smed,sscale,"S")
    panel=source_panel(sources); mainset=set(panel)
    mapping=[]
    for pop,uu in (("P_main",panel),("P_aux",[u for u in sources.canonical_uid if u not in mainset])):
        ordered=sorted(uu,key=lambda u:h("G_ORDER",u)); n=len(ordered)
        events=sources.set_index("canonical_uid").capture_event_id.to_dict()
        counts=[sum(events[ordered[i]]!=events[ordered[(i+s)%n]] for i in range(n)) for s in range(1,n)]
        shift=1+int(np.argmax(counts)) if n>1 else 0
        mapping.extend({"uid":uid,"donor_uid":ordered[(i+shift)%n],"population":pop,"shift":shift,
            "different_event":events[uid]!=events[ordered[(i+shift)%n]]} for i,uid in enumerate(ordered))
    write_json(POLICY/"SOURCE_G_DONOR_MAPPING.json",mapping)
    write_json(POLICY/"SOURCE_MAIN_PANEL.json",panel)
    for seed in (45,46):
        directory=POLICY/f"s{seed}";directory.mkdir(exist_ok=True)
        ss=[json.loads(s) for s in (DATA/f"SCHEDULES/s{seed}_source.jsonl").read_text().splitlines()]
        sj,sc=source_jobs(sources,sctx,seed,ss)
        write_json(directory/"S_JOBS.json",sj)
        pd.DataFrame(sc).to_parquet(directory/"S_CANDIDATE_LEDGER.parquet",index=False)
        ts=[json.loads(s) for s in (DATA/f"SCHEDULES/s{seed}_target.jsonl").read_text().splitlines()]
        candidates=[];workers=[];audits=[]
        for epoch in range(72,101):
            rows=[r for r in ts if r["epoch"]==epoch]
            visits=[{"epoch":epoch,"target_visit_ordinal":r["batch"]*8+p,"schedule_uid":uid}
                for r in rows for p,uid in enumerate(r["uids"])]
            route.EXPECTED_VISITS_PER_EPOCH=len(visits)
            worker,deltas,audit=_native_epoch(epoch,pd.DataFrame(visits),pctx,seed)
            candidates.append(deltas);workers.append(worker);audits.append(audit)
        cc=pd.concat(candidates,ignore_index=True)
        pd.concat(workers,ignore_index=True).to_parquet(directory/"P_NATIVE_ACTIVE_MASK.parquet",index=False)
        cc.to_parquet(directory/"P_VALID_CANDIDATES.parquet",index=False)
        jobs=[]
        for row in targets.to_dict("records"):
            uid=row["canonical_uid"]; valid=cc[cc.target_uid==uid].sort_values(["epoch","target_visit_ordinal"],kind="mergesort")
            selected=None
            if len(valid):
                rec=valid.iloc[selector_digest(uid)[2]%len(valid)]
                selected={"epoch":int(rec.epoch),"ordinal":int(rec.target_visit_ordinal),"dose":float(rec.intended_geometric_dose),
                    "destination_event":str(rec.destination_event_id),"delta_sha256":str(rec.native_delta_sha256),
                    "delta":[float(rec["native_delta__"+n]) for n in core.BASE],
                    "target_std":[float(rec["native_target_std__"+n]) for n in core.BASE]}
            jobs.append(make_job(row,pctx,seed,"P",selected,len(valid)))
        write_json(directory/"P_JOBS.json",jobs);write_json(directory/"P_NATIVE_EPOCH_AUDITS.json",audits)
    write_json(CONTRACT/"POLICY_SELECTION_FREEZE.json",{"utc":now(),"dataset_freeze_sha256":sha(CONTRACT/"DATASET_FREEZE.json"),
        "dataset_artifact_manifest_sha256":manifest_sha(CONTRACT),
        "ours_method_sha256":sha(CONTRACT/"OURS_METHOD_FREEZE.json"),"descriptor_rows_sha256":sha(POLICY/"DESCRIPTOR_ROWS.csv"),
        "files":[{"path":str(p),"sha256":sha(p)} for p in sorted(POLICY.rglob("*")) if p.is_file()],
        "seeds":[45,46],"P_Normalization_domains":["source_train","target_adapt"],"P_support_domain":"target_adapt",
        "S_Normalization_and_support":"source_train_only","no_animal_labels_or_predictions":True,
        "G_recipient_reoptimization":False,"no_new_tau_or_solver_rule":True})


def solve_worker(job):
    worker_start=time.perf_counter()
    cpu_mode()
    np,_,core,_=imports()
    import cv2
    cv2.setNumThreads(1)
    directory=POLICY/f"s{job['seed']}"/job["view"]
    directory.mkdir(exist_ok=True)
    receipt=directory/(job["uid"]+".json")
    if receipt.exists():
        result=read_json(receipt);assert result["task_key"]==job["task_key"]
        if result["status"]=="COMPLETE" and result.get("cache_path"):assert sha(result["cache_path"])==result["cache_sha256"]
        return result
    selected=job["selected_candidate"]
    result={"uid":job["uid"],"task_key":job["task_key"],"seed":job["seed"],"view":job["view"],"utc":now()}
    if selected is None:
        result.update(status="IDENTITY_NO_VALID_DELTA",theta=None)
    else:
        assert sha(job["image_path"])==job["image_sha256"]
        rgb,decoder=core.decode_q2_rgb(job["image_path"])
        palette=core.PaletteImage.from_rgb_uint8(rgb)
        with np.load(job["geometry_file"],allow_pickle=False) as g:
            diag=core.realization_case(palette=palette,uid=job["uid"],event_id=job["event_id"],case_type=job["view"],
                input_image_sha256=job["image_sha256"],decoder=decoder,original_std=np.asarray(job["original_std"]),
                target_std=np.asarray(selected["target_std"]),intended_dose=selected["dose"],target_delta_sha256=selected["delta_sha256"],
                donor_uid=job["uid"],donor_event_id=selected["destination_event"],median=g["median"],scale=g["scale"],event_std=g["event_std"])
            diag["legacy_CCT_endpoint_in_tube_diagnostic_unused"]=diag.pop("actual_endpoint_in_tube")
            diag["actual_endpoint_in_CICT_tube"]=bool(diag["actual_support_deviation"]<=float(g["tau"]))
        theta=np.asarray([diag["delta_t"],diag["c_t"],diag["s_c"]],dtype="<f8")
        continuous,_,_,_=palette.continuous_output(theta)
        deployed=np.rint(continuous*255.).astype(np.uint8)
        output=np.ascontiguousarray(deployed[palette.inverse_flat].reshape(rgb.shape))
        formal=hashlib.sha256(core.ARRAY_HASH_PREFIX+b"uint8"+str(tuple(rgb.shape)).encode("ascii")+
            np.ascontiguousarray(output[:,:,::-1]).tobytes()).hexdigest()
        assert formal==diag["output_array_sha256"],"Frozen solver/renderer byte mismatch"
        result.update(status="COMPLETE",theta=theta.tolist(),theta_f8_hex=theta.tobytes().hex(),
            pixel_sha256=hashlib.sha256(output.tobytes()).hexdigest(),formal_BGR_array_sha256=formal,diagnostics=diag)
        if job["view"]=="P":
            cache=directory/(job["uid"]+".png")
            assert cv2.imwrite(str(cache),output[:,:,::-1],[cv2.IMWRITE_PNG_COMPRESSION,1])
            replay=cv2.imread(str(cache))[:,:,::-1]
            assert np.array_equal(output,replay),"Lossless cache mismatch"
            result.update(cache_path=str(cache),cache_sha256=sha(cache))
        else:
            # Reuse S parameters for G without writing S images.
            result.update(S_pixels_evaluated_and_hashed=True,S_png_not_needed_by_any_registered_arm=True)
    import psutil
    memory=psutil.Process().memory_info()
    result["worker_peak_RSS_bytes"]=int(getattr(memory,"peak_wset",memory.rss))
    result["worker_total_wall_seconds"]=time.perf_counter()-worker_start
    write_json(receipt,result)
    return result


def render_g_worker(args):
    row,donor,seed=args
    cpu_mode()
    np,_,core,_=imports()
    import cv2
    from m1_ss2_seed45_parameter_adapter import render_rgb
    cv2.setNumThreads(1)
    directory=POLICY/f"s{seed}/G";directory.mkdir(exist_ok=True)
    receipt=directory/(row["canonical_uid"]+".json")
    if receipt.exists():
        rec=read_json(receipt)
        if rec["status"]=="COMPLETE":assert sha(rec["cache_path"])==rec["cache_sha256"]
        assert rec["donor_task_key"]==donor["task_key"]
        return rec
    rec={"uid":row["canonical_uid"],"seed":seed,"view":"G","donor_uid":donor["uid"],"donor_task_key":donor["task_key"],
        "status":donor["status"],"theta":donor["theta"],"recipient_reoptimization":False,"utc":now()}
    if donor["status"]=="COMPLETE":
        assert sha(row["image_path"])==row["image_sha256"]
        rgb,_=core.decode_q2_rgb(row["image_path"])
        output=render_rgb(rgb,donor["theta_f8_hex"])
        cache=directory/(row["canonical_uid"]+".png")
        assert cv2.imwrite(str(cache),output[:,:,::-1],[cv2.IMWRITE_PNG_COMPRESSION,1])
        assert np.array_equal(output,cv2.imread(str(cache))[:,:,::-1])
        rec.update(cache_path=str(cache),cache_sha256=sha(cache),theta_f8_hex=donor["theta_f8_hex"],
            pixel_sha256=hashlib.sha256(output.tobytes()).hexdigest())
    write_json(receipt,rec)
    return rec


def solve_all():
    np,pd,_,_=imports()
    import psutil
    assert (CONTRACT/"POLICY_SELECTION_FREEZE.json").exists()
    start=time.perf_counter()
    cap=cpu_cap_seconds()
    assert reuse_frozen_policy_assets()
    jobs=[j for seed in (45,46) for view in ("S","P") for j in read_json(POLICY/f"s{seed}/{view}_JOBS.json")]
    # Frozen pilot: 16 source and 16 adapt images.
    refs=[]
    for view in ("S","P"):
        refs.extend(sorted([j for j in jobs if j["seed"]==45 and j["view"]==view and j["selected_candidate"]],
            key=lambda j:hashlib.sha256(("CICT_CPU_FORECAST|"+j["uid"]).encode()).hexdigest())[:16])
    bench=[]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
        t0=time.perf_counter()
        bench=list(pool.map(solve_worker,refs,chunksize=1))
        elapsed=time.perf_counter()-t0
    pending=[j for j in jobs if not (POLICY/f"s{j['seed']}/{j['view']}/{j['uid']}.json").exists()]
    expensive=sum(j["selected_candidate"] is not None for j in pending)
    retained_two_worker_cost=sum(r.get("worker_total_wall_seconds",r.get("diagnostics",{}).get("elapsed_seconds",0)) for r in bench)/2
    benchmark_cost=max(elapsed,retained_two_worker_cost)
    per=benchmark_cost/max(len(refs),1)
    # Increase workers only after the resource check.
    workers=2
    prior=float(os.environ.get("CICT_CPU_PRIOR_SECONDS",0))
    forecast=prior+(time.perf_counter()-start)+per*expensive+len(image_rows())*.15
    scaling=[]
    # Reuse the measured pilot time.
    old_plan,old_scaling=retained_measurements()
    assert old_plan["reference_task_keys"]==[j["task_key"] for j in refs]
    selected_workers=old_scaling["selected_workers"]
    retained_trial=next(t for t in old_scaling["trials"]
        if t["workers"]==selected_workers and t["status"]=="MEASURED")
    job_index={j["task_key"]:j for j in jobs}
    assert len(retained_trial["pilot_task_keys"])==retained_trial["pilot_images"]
    retained_pilots=[]
    for key in retained_trial["pilot_task_keys"]:
        job=job_index[key]
        path=POLICY/f"s{job['seed']}/{job['view']}/{job['uid']}.json"
        assert path.exists(),"STOP_MISSING_RETAINED_PILOT"
        retained_pilots.append(solve_worker(job))
    peak=max(r["worker_peak_RSS_bytes"] for r in bench+retained_pilots)
    available=psutil.virtual_memory().available
    retained_resource_pass=(psutil.cpu_count(logical=False) or 1)>=selected_workers and available>8*1024**3+selected_workers*peak*1.25
    reuse={"utc":now(),"status":"RETAINED_PILOT_REUSED" if retained_resource_pass else "CURRENT_RESOURCE_GATE_REJECTED_REUSE",
        "original_scaling_sha256":sha(AUDIT/"CPU_SCALING_RESOURCE_GATE.json"),
        "selected_workers":selected_workers,"measured_worker_peak_bytes":peak,
        "current_available_RAM_bytes":available,"retained_pilot_task_keys":retained_trial["pilot_task_keys"],
        "retained_pilot_wall_seconds":retained_trial["wall_seconds"],"reserve_GiB":6,
        "future_GPU_process_RAM_provision_GiB":2,"resource_budget":budget_identity(),"same_frozen_jobs_and_solver":True}
    if retained_resource_pass:
        retained_per=retained_trial["wall_seconds"]/retained_trial["pilot_images"]
        if retained_per<per:
            workers=selected_workers;per=retained_per
        forecast=prior+(time.perf_counter()-start)+per*expensive+len(image_rows())*.15
    write_json(scaling_path(),reuse)
    if forecast>=cap:
        peak=max((r.get("worker_peak_RSS_bytes",0) for r in bench),default=0)
        # Reserve 6 GiB plus 2 GiB for source loading.
        trials=[]
        for candidate_workers in (4,3):
            if forecast<cap:break
            pre_pilot_available=psutil.virtual_memory().available
            can_scale=(psutil.cpu_count(logical=False) or 1)>=candidate_workers and peak>0 and pre_pilot_available>8*1024**3+candidate_workers*peak*1.25
            if not can_scale:
                trials.append({"workers":candidate_workers,"status":"INSUFFICIENT_MEASURED_RAM_OR_CORES",
                    "available_RAM_bytes":pre_pilot_available})
                continue
            pilot=sorted([j for j in pending if j["selected_candidate"]],key=lambda j:j["task_key"])[:16]
            t1=time.perf_counter()
            with ProcessPoolExecutor(max_workers=candidate_workers,mp_context=multiprocessing.get_context("spawn")) as pool:
                current=list(pool.map(solve_worker,pilot,chunksize=1))
                scaling.extend(current)
            pilot_wall=time.perf_counter()-t1
            pilot_per=pilot_wall/max(len(pilot),1)
            pending=[j for j in jobs if not (POLICY/f"s{j['seed']}/{j['view']}/{j['uid']}.json").exists()]
            expensive=sum(j["selected_candidate"] is not None for j in pending)
            if pilot_per<per:
                workers=candidate_workers;per=pilot_per
            forecast=prior+(time.perf_counter()-start)+per*expensive+len(image_rows())*.15
            trials.append({"workers":candidate_workers,"status":"MEASURED","pre_pilot_available_RAM":pre_pilot_available,
                "pilot_task_keys":[j["task_key"] for j in pilot],"pilot_images":len(pilot),"wall_seconds":pilot_wall,
                "per_image_wall_seconds":pilot_per,"forecast_after_pilot_seconds":forecast})
        write_json(scaling_path(),{"utc":now(),"measured_two_worker_peak_bytes":peak,"retained_measurement_check":reuse,
            "trials":trials,"selected_workers":workers,"reserve_GiB":6,"future_GPU_process_RAM_provision_GiB":2,
            "same_frozen_jobs_and_solver":True,"no_extra_unique_tasks":True})
    plan={"utc":now(),"reference_images":len(refs),"measured_two_worker_wall_seconds":elapsed,
        "retained_reference_two_worker_cost_seconds":retained_two_worker_cost,"cost_used_for_reference_seconds":benchmark_cost,
        "reference_task_keys":[j["task_key"] for j in refs],"cached_reference_does_not_become_free_remaining_forecast":True,
        "remaining_expensive_jobs":expensive,"predicted_active_remaining_seconds":per*expensive,
        "conservative_preparation_seconds":forecast,"cap_seconds":cap,"workers":workers,
        "available_RAM_bytes":psutil.virtual_memory().available,"reserve_RAM_bytes":6*1024**3,
        "no_solver_or_method_changes_for_budget":True,"scaled_pilot_unique_jobs":len(scaling),
        "prior_CPU_preparation_seconds":prior,"solver_code_sha256":sha(Path(__file__)),
        "resource_budget":budget_identity(),"prior_measured_pilot_receipts_reverified":len(retained_pilots)}
    write_json(forecast_path(),plan)
    assert forecast<cap,"STOP_FORECAST_CPU_BUDGET_NO_METHOD_SUBSTITUTION"
    assert psutil.virtual_memory().available>6*1024**3,"Insufficient RAM reserve"
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for offset in range(0,len(pending),32):
            list(pool.map(solve_worker,pending[offset:offset+32],chunksize=1))
            write_json(OUT/"METHOD_PREPARATION_STATE.json",{"utc":now(),"stage":"EXACT_POWELL",
                "processed_jobs":len(jobs)-len(pending)+min(offset+32,len(pending)),"total_jobs":len(jobs),
                "session_active_wall_seconds":time.perf_counter()-start,"workers":workers,"complete":False})
            assert prior+time.perf_counter()-start<cap,"STOP_CPU_PREPARATION_CAP"
            check_controller_stop("METHOD_EXACT_SOLVE_AND_CACHE")
    mapping=read_json(POLICY/"SOURCE_G_DONOR_MAPPING.json")
    rows={r["canonical_uid"]:r for r in read_json(POLICY/"DESCRIPTORS.json").values() if r["domain"]=="source_train"}
    for seed in (45,46):
        sr={j["uid"]:read_json(POLICY/f"s{seed}/S/{j['uid']}.json") for j in jobs if j["view"]=="S" and j["seed"]==seed}
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as pool:
            render_jobs=[(rows[m["uid"]],sr[m["donor_uid"]],seed) for m in mapping]
            for offset in range(0,len(render_jobs),32):
                list(pool.map(render_g_worker,render_jobs[offset:offset+32],chunksize=1))
                assert prior+time.perf_counter()-start<cap,"STOP_CPU_PREPARATION_CAP"
                check_controller_stop("METHOD_EXACT_SOLVE_AND_CACHE")
        lookup={}
        for label in ("P","G"):
            lookup[label]={p.stem:read_json(p) for p in (POLICY/f"s{seed}/{label}").glob("*.json")}
        write_json(POLICY/f"s{seed}/TRAINING_LOOKUP.json",lookup)
    manifest=[{"path":str(p),"sha256":sha(p)} for p in sorted(POLICY.rglob("*")) if p.is_file()]
    write_json(CONTRACT/"METHOD_ASSET_FREEZE.json",{"utc":now(),"status":"FROZEN_BEFORE_TAILS_AND_ORACLE_GT",
        "selection_freeze_sha256":sha(CONTRACT/"POLICY_SELECTION_FREEZE.json"),"files":manifest,
        "wall_seconds_solve_stage":time.perf_counter()-start,"seed_caches":[45,46],
        "Ours_Oracle_same_seed_same_assets":True,"Target_Dev_Test_reads":0,"Adapt_GT_reads":0})


if __name__=="__main__":
    cpu_mode()
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("descriptors","freeze-policies","solve"))
    args=parser.parse_args()
    {"descriptors":descriptors,"freeze-policies":freeze_policies,"solve":solve_all}[args.stage]()
