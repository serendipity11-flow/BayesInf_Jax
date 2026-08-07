"""Create a door-style standalone reporting bundle for off-road FlowSDE."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save(fig, root, stem):
    fig.tight_layout()
    fig.savefig(root / f"{stem}.png", dpi=180)
    fig.savefig(root / f"{stem}.pdf")
    plt.close(fig)


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((args.source / "evaluation/metrics.json").read_text())
    history = json.loads((args.source / "history.json").read_text())
    config = json.loads((args.source / "config.json").read_text())
    write_csv(args.output / "checkpoints.csv", history)

    final = {"method": "FlowSDE-RealNVP-Seg10-Energy0.1", "seed": config["seed"],
             "segments": config.get("train_segments"), "segment_length": config["seg_len"],
             "stride": config["stride"], "transition_weight": config["transition_weight"],
             "energy_weight": config["energy_weight"],
             **{k:v for k,v in metrics.items() if not isinstance(v,dict)}}
    write_csv(args.output / "final_results.csv", [final])
    state_rows = [{"state": name, "one_step_rmse": value,
                   "coverage_90": metrics["per_state_90pct_coverage"][name]}
                  for name,value in metrics["per_state_rmse"].items()]
    write_csv(args.output / "per_state_metrics.csv", state_rows)
    horizon_rows = [{"steps":int(h), "seconds":int(h)*config["dt"], "aggregate_rmse":v}
                    for h,v in metrics["horizon_aggregate_rmse"].items()]
    write_csv(args.output / "horizon_metrics.csv", horizon_rows)
    flags=[]
    if metrics["mean_90pct_coverage"] < .85:
        flags.append({"issue":"undercoverage", "observed":metrics["mean_90pct_coverage"], "required":.85})
    if metrics["recursive_particle_divergence_fraction_50m"] > .05:
        flags.append({"issue":"particle_divergence", "observed":metrics["recursive_particle_divergence_fraction_50m"], "required":.05})
    if metrics["recursive_final_xy_error_mean"] > 1:
        flags.append({"issue":"long_horizon_xy_error_m", "observed":metrics["recursive_final_xy_error_mean"], "required":1.0})
    write_csv(args.output / "flags.csv", flags)

    epoch=np.array([r["epoch"] for r in history])
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    for key,label in (("train_transition_nll","train"),("validation_transition_nll","validation")):
        axes[0].plot(epoch,[r[key] for r in history],label=label)
    axes[0].set(title="Transition NLL",xlabel="epoch");axes[0].legend();axes[0].grid(alpha=.25)
    for key,label in (("train_energy_score","train"),("validation_energy_score","validation")):
        axes[1].plot(epoch,[r[key] for r in history],label=label)
    axes[1].set(title="10-step energy score",xlabel="epoch");axes[1].legend();axes[1].grid(alpha=.25)
    axes[2].plot(epoch,[r["train_total"] for r in history],label="train")
    axes[2].plot(epoch,[r["validation_total"] for r in history],label="validation")
    axes[2].set(title="Weighted total objective",xlabel="epoch");axes[2].legend();axes[2].grid(alpha=.25)
    save(fig,args.output,"training_curves")

    names=[r["state"] for r in state_rows]; positions=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(positions,[r["one_step_rmse"] for r in state_rows],color="tab:blue")
    ax.set_xticks(positions,names,rotation=35,ha="right");ax.set(ylabel="RMSE",title="Held-out one-step RMSE by state");ax.grid(axis="y",alpha=.25)
    save(fig,args.output,"state_rmse")
    fig,ax=plt.subplots(figsize=(9,4.8));ax.bar(positions,[r["coverage_90"] for r in state_rows],color="tab:green")
    ax.axhline(.9,color="k",ls="--",label="nominal 90%");ax.set_xticks(positions,names,rotation=35,ha="right");ax.set(ylim=(0,1),ylabel="empirical coverage",title="Held-out 90% interval coverage");ax.legend();ax.grid(axis="y",alpha=.25)
    save(fig,args.output,"state_coverage")
    fig,ax=plt.subplots(figsize=(7,4.5));ax.plot([r["seconds"] for r in horizon_rows],[r["aggregate_rmse"] for r in horizon_rows],marker="o")
    ax.set(xlabel="rollout horizon [s]",ylabel="aggregate RMSE",title="Recursive rollout error");ax.grid(alpha=.25)
    save(fig,args.output,"rollout_horizon")

    with np.load(args.source/"evaluation/trajectory_data.npz") as d:
        truth,mean,particles,ids=d["truth"],d["recursive_mean"],d["recursive_particles"],d["test_episode_ids"]
    count=min(3,len(particles));fig,axes=plt.subplots(1,count,figsize=(5*count,4.5),squeeze=False)
    for i,ax in enumerate(axes[0]):
        visible=np.concatenate((truth[i,:,:2],mean[i,:,:2]));low,high=visible.min(0),visible.max(0);margin=np.maximum(.5,.08*(high-low))
        for p in particles[i]:ax.plot(p[:,0],p[:,1],color="tab:blue",alpha=.07,lw=.6)
        ax.plot(truth[i,:,0],truth[i,:,1],"k",lw=2,label="MuJoCo")
        ax.plot(mean[i,:,0],mean[i,:,1],color="tab:orange",lw=2,label="zero-noise")
        ax.set(title=f"held-out episode {ids[i]}",xlabel="x [m]",ylabel="y [m]",xlim=(low[0]-margin[0],high[0]+margin[0]),ylim=(low[1]-margin[1],high[1]+margin[1]));ax.legend();ax.grid(alpha=.2)
    save(fig,args.output,"trajectory_samples")
    shutil.copy2(args.source/"evaluation/metrics.json",args.output/"metrics.json")
    summary=("| Method | Test episodes | One-step RMSE | Energy score | 90% coverage | Final XY error | Divergence |\n"
             "|---|---:|---:|---:|---:|---:|---:|\n"
             f"| FlowSDE-RealNVP-Seg10-Energy0.1 | {metrics['test_episodes']} | {metrics['one_step_aggregate_rmse']:.3f} | {metrics['one_step_energy_score']:.3f} | {metrics['mean_90pct_coverage']:.3f} | {metrics['recursive_final_xy_error_mean']:.3f} m | {metrics['recursive_particle_divergence_fraction_50m']:.3f} |\n\n"
             "The model is numerically stable but severely under-covers the held-out transitions and accumulates large long-horizon position error.\n")
    (args.output/"results.md").write_text(summary)
    print(args.output)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,default=Path("results/results_optimal_control/offroad_flowsde/seg10_stride10_tw1_energy0p1_lr1e4_seed0_rerun"))
    p.add_argument("--output",type=Path,default=Path("results/results_optimal_control/offroad_flowsde/generative_dynamics_evaluation"))
    return p
if __name__=="__main__":run(parser().parse_args())
