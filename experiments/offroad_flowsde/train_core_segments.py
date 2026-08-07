"""Train FlowSDE on a 14-D core while reconstructing XY kinematically."""

import argparse,json,time
from pathlib import Path
import equinox as eqx
import jax,jax.numpy as jnp,jax.random as jr
import numpy as np,optax
from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE
from .structured_core import reconstruct_next


def split_ids(e,seed):
    ids=np.random.default_rng(seed).permutation(np.unique(e));n=max(1,round(.1*len(ids)))
    return ids[2*n:],ids[n:2*n],ids[:n]


def windows(e,ids,length,stride):
    out=[]
    for episode in ids:
        idx=np.flatnonzero(e==episode)
        out.extend(idx[s:s+length] for s in range(0,len(idx)-length+1,stride))
    return np.stack(out)


def train(args):
    with np.load(args.data) as d:
        x,u,xn,e=(d[k].copy() for k in ("observation.state","action","next_observation.state","episode_index"))
    if x.shape[1:]!=(16,) or u.shape[1:]!=(2,):raise ValueError("expected 16-D context and 2-D action")
    train_ids,val_ids,test_ids=split_ids(e,args.seed); mask=np.isin(e,train_ids)
    core_delta=xn[:,2:]-x[:,2:]
    stats={"context_mean":x[mask].mean(0),"context_std":x[mask].std(0)+1e-6,
           "action_mean":u[mask].mean(0),"action_std":u[mask].std(0)+1e-6,
           "core_delta_mean":core_delta[mask].mean(0),"core_delta_std":core_delta[mask].std(0)+1e-6}
    cx=(x-stats["context_mean"])/stats["context_std"]
    cu=(u-stats["action_mean"])/stats["action_std"]
    cd=(core_delta-stats["core_delta_mean"])/stats["core_delta_std"]
    cy=(xn-stats["context_mean"])/stats["context_std"]
    tw,vw=windows(e,train_ids,args.seg_len,args.stride),windows(e,val_ids,args.seg_len,args.stride)
    model=FlowSDE(16,2,14,jr.PRNGKey(args.seed),dt=args.dt,width=args.width,depth=args.depth,
                  condition_dim=args.condition_dim,flow_layers=args.flow_layers)
    optimizer=optax.chain(optax.clip_by_global_norm(args.grad_clip),optax.adamw(args.lr,weight_decay=1e-5))
    opt_state=optimizer.init(eqx.filter(model,eqx.is_inexact_array))
    cm,cs,dm,ds=map(jnp.asarray,(stats["context_mean"],stats["context_std"],
                                stats["core_delta_mean"],stats["core_delta_std"]))

    def path(candidate,initial_n,actions_n,keys,stochastic):
        def step(current_n,item):
            action_n,key=item; delta_n=(candidate.step(current_n,action_n,key,True) if stochastic
                                        else candidate.mean_delta(current_n,action_n))
            current=current_n*cs+cm; next_core=current[2:]+delta_n*ds+dm
            nxt=reconstruct_next(current,next_core,args.dt); next_n=(nxt-cm)/cs
            return next_n,next_n
        return jax.lax.scan(step,initial_n,(actions_n,keys))[1]

    def objective(candidate,bx,bu,bd,by,keys):
        transition=-jnp.mean(jax.vmap(jax.vmap(candidate.log_prob))(bd,bx,bu))
        drift=jnp.mean((jax.vmap(jax.vmap(candidate.mean_delta))(bx,bu)-bd)**2)
        det=jax.vmap(path,in_axes=(None,0,0,0,None))(candidate,bx[:,0],bu,keys[:,0],False)
        rollout=jnp.mean((det-by)**2)
        def score(initial,actions,truth,k):
            samples=jax.vmap(path,in_axes=(None,None,None,0,None))(candidate,initial,actions,k,True)
            sf=samples.reshape(args.particles,-1);tf=truth.reshape(-1)
            return (jnp.mean(jnp.linalg.norm(sf-tf,axis=-1))-.5*jnp.mean(
                jnp.linalg.norm(sf-jnp.roll(sf,1,axis=0),axis=-1)))/jnp.sqrt(tf.size)
        energy=jnp.mean(jax.vmap(score)(bx[:,0],bu,by,keys))
        total=(args.transition_weight*transition+args.drift_weight*drift+
               args.rollout_weight*rollout+args.energy_weight*energy)
        return total,(transition,drift,rollout,energy)

    @eqx.filter_jit
    def update(m,o,bx,bu,bd,by,k):
        (loss,parts),g=eqx.filter_value_and_grad(objective,has_aux=True)(m,bx,bu,bd,by,k)
        updates,o=optimizer.update(g,o,eqx.filter(m,eqx.is_inexact_array))
        return eqx.apply_updates(m,updates),o,loss,parts

    best,best_val,history=model,float("inf"),[];rng=np.random.default_rng(args.seed);key=jr.PRNGKey(args.seed+1);started=time.time()
    for epoch in range(1,args.epochs+1):
        values=[];order=rng.permutation(len(tw))
        for start in range(0,len(tw),args.batch_size):
            wi=tw[order[start:start+args.batch_size]]
            key,bk=jr.split(key);keys=jr.split(bk,len(wi)*args.particles*args.seg_len).reshape(len(wi),args.particles,args.seg_len,2)
            model,opt_state,loss,parts=update(model,opt_state,jnp.asarray(cx[wi]),jnp.asarray(cu[wi]),jnp.asarray(cd[wi]),jnp.asarray(cy[wi]),keys)
            values.append([float(loss),*[float(p) for p in parts]])
        vi=vw[:min(args.validation_segments,len(vw))];key,vk=jr.split(key)
        keys=jr.split(vk,len(vi)*args.particles*args.seg_len).reshape(len(vi),args.particles,args.seg_len,2)
        val,vparts=objective(model,jnp.asarray(cx[vi]),jnp.asarray(cu[vi]),jnp.asarray(cd[vi]),jnp.asarray(cy[vi]),keys)
        avg=np.mean(values,0);row={"epoch":epoch,"train_total":avg[0],"train_nll":avg[1],"train_drift_mse":avg[2],"train_rollout_mse":avg[3],"train_energy":avg[4],"validation_total":float(val),"validation_parts":[float(p) for p in vparts]}
        history.append(row);print(json.dumps(row),flush=True)
        if float(val)<best_val:best,best_val=model,float(val)
    args.output.mkdir(parents=True,exist_ok=True);eqx.tree_serialise_leaves(args.output/"flowsde_core.eqx",best)
    np.savez(args.output/"normalization.npz",**stats)
    cfg={**vars(args),"data":str(args.data),"output":str(args.output),"learned_core_dim":14,
         "xy_update":"kinematic trapezoidal integration","train_episodes":train_ids.tolist(),
         "validation_episodes":val_ids.tolist(),"test_episodes":test_ids.tolist(),"best_validation_total":best_val,"elapsed_seconds":time.time()-started}
    (args.output/"config.json").write_text(json.dumps(cfg,indent=2,default=str)+"\n");(args.output/"history.json").write_text(json.dumps(history,indent=2)+"\n")


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seg-len",type=int,default=10);p.add_argument("--stride",type=int,default=10);p.add_argument("--particles",type=int,default=4)
    p.add_argument("--transition-weight",type=float,default=1);p.add_argument("--drift-weight",type=float,default=5);p.add_argument("--rollout-weight",type=float,default=2);p.add_argument("--energy-weight",type=float,default=.1)
    p.add_argument("--epochs",type=int,default=100);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--validation-segments",type=int,default=1024)
    p.add_argument("--lr",type=float,default=1e-4);p.add_argument("--grad-clip",type=float,default=1);p.add_argument("--dt",type=float,default=.05);p.add_argument("--width",type=int,default=192);p.add_argument("--depth",type=int,default=3);p.add_argument("--condition-dim",type=int,default=128);p.add_argument("--flow-layers",type=int,default=4);p.add_argument("--seed",type=int,default=0);return p
if __name__=="__main__":train(parser().parse_args())
