"""Held-out evaluation for the continuous-yaw segment-trained FlowSDE."""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE

NAMES = ("x", "y", "psi", "v_long", "v_lat", "yaw_rate", "wheel_slip")


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((args.result / "config.json").read_text())
    model = FlowSDE(7, 2, 7, jr.PRNGKey(cfg["seed"]), dt=cfg["dt"], width=cfg["width"],
                    depth=cfg["depth"], condition_dim=cfg["condition_dim"], flow_layers=cfg["flow_layers"])
    model = eqx.tree_deserialise_leaves(args.result / "flowsde_seg10.eqx", model)
    stats = dict(np.load(args.result / "normalization.npz"))
    with np.load(args.data) as data:
        x, u, xn, episode = (data[k] for k in
            ("observation.state", "action", "next_observation.state", "episode_index"))
    mask = np.isin(episode, cfg["test_episodes"]); xh, uh, xnh = x[mask], u[mask], xn[mask]
    xs = (xh-stats["state_mean"])/stats["state_std"]; us = (uh-stats["action_mean"])/stats["action_std"]
    ds = ((xnh-xh)-stats["delta_mean"])/stats["delta_std"]
    mean_dn = np.asarray(jax.jit(jax.vmap(model.mean_delta))(jnp.asarray(xs), jnp.asarray(us)))
    pred = xh + mean_dn*stats["delta_std"] + stats["delta_mean"]
    error = pred-xnh; rmse = np.sqrt(np.mean(error**2, axis=0))
    nll = float(-np.mean(np.asarray(jax.jit(jax.vmap(model.log_prob))(
        jnp.asarray(ds), jnp.asarray(xs), jnp.asarray(us)))))

    rng = np.random.default_rng(args.seed); ci = rng.choice(len(xh), min(args.calibration_samples, len(xh)), False)
    keys = jr.split(jr.PRNGKey(args.seed), len(ci)*args.particles).reshape(len(ci), args.particles, 2)
    sample = jax.jit(jax.vmap(jax.vmap(lambda s, a, k: model.step(s, a, k, True),
                                       in_axes=(None, None, 0)), in_axes=(0, 0, 0)))
    sample_dn = np.asarray(sample(jnp.asarray(xs[ci]), jnp.asarray(us[ci]), keys))
    sample_next = xh[ci, None] + sample_dn*stats["delta_std"] + stats["delta_mean"]
    lo, hi = np.quantile(sample_next, (.05, .95), axis=1)
    coverage = np.mean((xnh[ci]>=lo)&(xnh[ci]<=hi), axis=0)
    energy = float(np.mean(np.linalg.norm(sample_next-xnh[ci,None], axis=-1))
                   - .5*np.mean(np.linalg.norm(sample_next-np.roll(sample_next,1,axis=1), axis=-1)))

    ids=np.asarray(cfg["test_episodes"]); horizon=min(args.horizon,min(np.sum(episode==e) for e in ids))
    truth=[]; controls=[]
    for e in ids:
        idx=np.flatnonzero(episode==e)[:horizon]
        truth.append(np.concatenate((x[idx[:1]],xn[idx]),axis=0)); controls.append(u[idx])
    truth,controls=np.stack(truth),np.stack(controls)
    sm,ss,dm,dsd=(jnp.asarray(stats[k]) for k in ("state_mean","state_std","delta_mean","delta_std"))
    def rollout(initial, commands, keys=None):
        def step(current, item):
            command=item if keys is None else item[0]; key=None if keys is None else item[1]
            sn=(current-sm)/ss; un=(command-jnp.asarray(stats["action_mean"]))/jnp.asarray(stats["action_std"])
            dn=model.mean_delta(sn,un) if key is None else model.step(sn,un,key,True)
            nxt=current+dn*dsd+dm
            return nxt,nxt
        items=commands if keys is None else (commands,keys)
        return jax.lax.scan(step,initial,items)[1]
    mean=np.asarray(jax.jit(jax.vmap(rollout,in_axes=(0,0,None)))(jnp.asarray(truth[:,0]),jnp.asarray(controls),None))
    mean=np.concatenate((truth[:,:1],mean),axis=1)
    horizon_rmse=np.sqrt(np.mean((mean-truth)**2,axis=(0,2))); xy=np.linalg.norm(mean[...,:2]-truth[...,:2],axis=-1)
    count=min(args.plot_episodes,len(ids)); pkeys=jr.split(jr.PRNGKey(args.seed+1),count*args.particles*horizon)
    pkeys=pkeys.reshape(count,args.particles,horizon,2)
    particle_fn=jax.jit(jax.vmap(jax.vmap(rollout,in_axes=(None,None,0)),in_axes=(0,0,0)))
    particles=np.asarray(particle_fn(jnp.asarray(truth[:count,0]),jnp.asarray(controls[:count]),pkeys))
    particles=np.concatenate((np.broadcast_to(truth[:count,None,:1],(count,args.particles,1,7)),particles),axis=2)
    divergence=np.max(np.linalg.norm(particles[...,:2],axis=-1),axis=-1)>50

    metrics={"test_episodes":len(ids),"test_transitions":len(xh),"test_normalized_nll":nll,
      "one_step_aggregate_rmse":float(np.sqrt(np.mean(error**2))),"one_step_energy_score":energy,
      "mean_90pct_coverage":float(np.mean(coverage)),"per_state_rmse":dict(zip(NAMES,map(float,rmse))),
      "per_state_90pct_coverage":dict(zip(NAMES,map(float,coverage))),
      "recursive_final_xy_error_mean":float(np.mean(xy[:,-1])),
      "recursive_final_xy_error_median":float(np.median(xy[:,-1])),
      "recursive_particle_divergence_fraction_50m":float(np.mean(divergence)),
      "horizon_aggregate_rmse":{str(h):float(horizon_rmse[h]) for h in (10,25,50,100) if h<=horizon}}
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    np.savez_compressed(args.output/"trajectory_data.npz",truth=truth,recursive_mean=mean,
                        recursive_particles=particles,horizon_rmse=horizon_rmse,test_episode_ids=ids)

    hist=json.loads((args.result/"history.json").read_text()); ep=np.array([r["epoch"] for r in hist])
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].plot(ep,[r["train_transition_nll"] for r in hist],label="train");ax[0].plot(ep,[r["validation_transition_nll"] for r in hist],label="validation");ax[0].set(title="Transition NLL",xlabel="epoch");ax[0].legend()
    ax[1].plot(ep,[r["train_energy_score"] for r in hist],label="train");ax[1].plot(ep,[r["validation_energy_score"] for r in hist],label="validation");ax[1].set(title="10-step energy score",xlabel="epoch");ax[1].legend()
    ax[2].bar(np.arange(7)-.2,rmse,.4,label="RMSE"); twin=ax[2].twinx();twin.bar(np.arange(7)+.2,coverage,.4,color="tab:green",label="90% coverage");twin.axhline(.9,color="k",ls="--");ax[2].set_xticks(range(7),NAMES,rotation=45,ha="right");ax[2].set_title("Held-out one-step metrics");ax[2].legend(loc="upper left");twin.legend(loc="upper right")
    fig.tight_layout();fig.savefig(args.output/"performance_metrics.png",dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,count,figsize=(5*count,4.5),squeeze=False)
    for i,a in enumerate(axes[0]):
        visible=np.concatenate((truth[i,:,:2],mean[i,:,:2])); low,high=visible.min(0),visible.max(0);margin=np.maximum(.5,.08*(high-low))
        a.plot(truth[i,:,0],truth[i,:,1],"k",lw=2,label="MuJoCo")
        for p in particles[i]:a.plot(p[:,0],p[:,1],color="tab:blue",alpha=.08,lw=.6)
        a.plot(mean[i,:,0],mean[i,:,1],color="tab:orange",lw=2,label="FlowSDE mean");a.set(xlabel="x [m]",ylabel="y [m]",title=f"held-out episode {ids[i]}",xlim=(low[0]-margin[0],high[0]+margin[0]),ylim=(low[1]-margin[1],high[1]+margin[1]));a.legend()
    fig.tight_layout();fig.savefig(args.output/"trajectories.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(np.arange(horizon+1)*cfg["dt"],horizon_rmse);ax.grid(alpha=.3);ax.set(xlabel="horizon [s]",ylabel="aggregate state RMSE",title="Recursive mean rollout error");fig.tight_layout();fig.savefig(args.output/"horizon_rmse.png",dpi=180);plt.close(fig)
    print(json.dumps(metrics,indent=2))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/offroad_flowsde/transitions_psi.npz"))
    p.add_argument("--result",type=Path,default=Path("results/results_optimal_control/offroad_flowsde/seg10_stride10_tw1_energy0p1_seed0"))
    p.add_argument("--output",type=Path,default=Path("results/results_optimal_control/offroad_flowsde/seg10_stride10_tw1_energy0p1_seed0/evaluation"))
    p.add_argument("--horizon",type=int,default=100);p.add_argument("--particles",type=int,default=64);p.add_argument("--calibration-samples",type=int,default=5000);p.add_argument("--plot-episodes",type=int,default=3);p.add_argument("--seed",type=int,default=2026)
    return p
if __name__=="__main__":run(parser().parse_args())
