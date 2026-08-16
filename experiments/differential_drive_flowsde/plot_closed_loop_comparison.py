"""Create closed-loop performance tables from completed complex-navigation runs."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    flow = json.loads(args.flowsde.read_text()); physical = json.loads(args.physical.read_text())
    flow_optimizer = "MPPI" if "MPPI" in flow["controller"] else "CEM"
    specifications = [(f"FlowSDE + {flow_optimizer}", flow, args.flowsde_hardware,
                       flow_optimizer, flow["candidates"], "synchronous"),
                      ("Physical + MPPI", physical, args.physical_hardware, "MPPI",
                       physical["samples"], "synchronous")]
    async_flow = json.loads(args.async_flowsde.read_text()) if args.async_flowsde else None
    async_physical = json.loads(args.async_physical.read_text()) if args.async_physical else None
    if async_flow:
        specifications.append((f"FlowSDE + {flow_optimizer} (async)", async_flow,
                               args.flowsde_hardware, flow_optimizer,
                               async_flow["candidates"], "asynchronous"))
    if async_physical:
        specifications.append(("Physical + MPPI (async)", async_physical,
                               args.physical_hardware, "MPPI",
                               async_physical["samples"], "asynchronous"))
    aggregate = []
    for label, metrics, hardware, optimizer, candidates, execution in specifications:
        episodes = metrics["episodes"]
        aggregate.append({"controller": label, "execution": execution,
                          "hardware": hardware, "optimizer": optimizer,
                          "candidates": candidates, "horizon": metrics["horizon"],
                          "stochastic_samples": metrics["stochastic_samples_per_candidate"],
                          "success_rate_percent": 100 * metrics["success_rate"],
                          "waypoints_completed": sum(e["completed_waypoints"] for e in episodes),
                          "waypoints_total": sum(len(e["waypoints"]) for e in episodes),
                          "mean_steps": float(np.mean([e["steps"] for e in episodes])),
                          "mean_navigation_time_s": .02 * float(np.mean([e["steps"] for e in episodes])),
                          "mean_final_distance_m": metrics["mean_final_distance_m"],
                          "mean_planning_ms": metrics["mean_planning_ms"],
                          "p95_planning_ms": metrics["p95_planning_ms"],
                          "planner_frequency_hz": metrics.get(
                              "planner_frequency_hz", metrics["control_frequency_hz"]),
                          "command_frequency_hz": metrics.get(
                              "command_frequency_hz", metrics["control_frequency_hz"]),
                          "mean_latency_compensation_steps": metrics.get(
                              "mean_latency_compensation_steps", 0.0)})
    per_task = []
    for task in range(3):
        f, p = flow["episodes"][task], physical["episodes"][task]
        per_task.append({"task": task, "FlowSDE success": f["success"], "Physical success": p["success"],
                         "FlowSDE steps": f["steps"], "Physical steps": p["steps"],
                         "FlowSDE final distance m": f["final_distance_m"],
                         "Physical final distance m": p["final_distance_m"],
                         "FlowSDE planning ms": f["mean_planning_ms"],
                         "Physical planning ms": p["mean_planning_ms"]})
        if async_flow and async_physical:
            af, ap = async_flow["episodes"][task], async_physical["episodes"][task]
            per_task[-1].update({"Async FlowSDE success": af["success"],
                                 "Async Physical success": ap["success"],
                                 "Async FlowSDE steps": af["steps"],
                                 "Async Physical steps": ap["steps"],
                                 "Async FlowSDE final distance m": af["final_distance_m"],
                                 "Async Physical final distance m": ap["final_distance_m"],
                                 "Async FlowSDE planning ms": af["mean_planning_ms"],
                                 "Async Physical planning ms": ap["mean_planning_ms"],
                                 "Async FlowSDE compensation steps": af["mean_latency_compensation_steps"],
                                 "Async Physical compensation steps": ap["mean_latency_compensation_steps"]})
    matched = args.flowsde_hardware == args.physical_hardware
    payload = {"timing_note": (f"Hardware matched: both evaluations used {args.flowsde_hardware}."
                                if matched else
                                f"Hardware not matched: FlowSDE used {args.flowsde_hardware}; physical MPPI used {args.physical_hardware}."),
               "aggregate": aggregate, "per_task": per_task}
    (args.output / "closed_loop_performance_comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    for filename, records in (("closed_loop_performance_comparison.csv", aggregate),
                              ("closed_loop_per_task_comparison.csv", per_task)):
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0]); writer.writeheader(); writer.writerows(records)

    headers = ("Controller", "Success", "Waypoints", "Mean\nsteps", "Mean nav.\ntime [s]",
               "Final dist.\n[m]", "Mean plan.\n[ms]", "P95 plan.\n[ms]",
               "Planner\n[Hz]", "Command\n[Hz]", "Latency\nsteps")
    cells = [[r["controller"], f'{r["success_rate_percent"]:.0f}%',
              f'{r["waypoints_completed"]}/{r["waypoints_total"]}', f'{r["mean_steps"]:.1f}',
              f'{r["mean_navigation_time_s"]:.2f}', f'{r["mean_final_distance_m"]:.3f}',
              f'{r["mean_planning_ms"]:.2f}', f'{r["p95_planning_ms"]:.2f}',
              f'{r["planner_frequency_hz"]:.2f}', f'{r["command_frequency_hz"]:.2f}',
              f'{r["mean_latency_compensation_steps"]:.1f}'] for r in aggregate]
    figure, ax = plt.subplots(figsize=(19, 5.4)); ax.axis("off")
    table = ax.table(cellText=cells, colLabels=headers, cellLoc="center", loc="center",
                     colWidths=(.17, .075, .085, .08, .09, .09, .09, .09, .075, .075, .07))
    table.auto_set_font_size(False); table.set_fontsize(13); table.scale(1, 2.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0: cell.set_text_props(weight="bold", fontsize=13); cell.set_facecolor("#dce6f1")
    ax.set_title("Complex navigation: closed-loop performance", fontsize=20, pad=18)
    figure.text(.5, .045, payload["timing_note"],
                ha="center", fontsize=12, style="italic")
    figure.tight_layout(rect=(0, .08, 1, 1))
    figure.savefig(args.output / "closed_loop_performance_comparison.pdf", bbox_inches="tight")
    figure.savefig(args.output / "closed_loop_performance_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(figure); print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowsde", type=Path, required=True); parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flowsde-hardware", default="A30 GPU")
    parser.add_argument("--physical-hardware", default="CPU (4 cores)")
    parser.add_argument("--async-flowsde", type=Path)
    parser.add_argument("--async-physical", type=Path)
    run(parser.parse_args())
