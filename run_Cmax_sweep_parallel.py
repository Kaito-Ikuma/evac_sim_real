#!/usr/bin/env python3
"""
Run a C_max sweep for the adaptive theory -> coarse -> fine simulator.

This launcher starts one MPI job per C_max value and limits the number of
concurrent jobs. It is intended to be executed inside an already allocated
interactive/batch job on SQUID or another HPC system.

Default sweep:
  N = 2000
  mu_scale = 1.0
  C_max = 20, 30, 50, 75, 100

Important:
  If np_per_run=16 and max_parallel=4, this uses up to 64 MPI ranks at once.
  If you set max_parallel=5, it uses 80 MPI ranks, so make sure your allocation
  has at least 80 physical cores or ranks available.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path


def fmt_param(x: float | int) -> str:
    xf = float(x)
    if abs(xf - int(xf)) < 1e-12:
        return str(int(xf))
    return f"{xf:.4g}".replace(".", "p").replace("-", "m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--script", default="simulate_adaptive_cmax_sweep_param.py")
    p.add_argument("--run_root", default="simulation_results_adaptive_sweep_cmax")
    p.add_argument("--cmax_values", nargs="+", type=float, default=[20, 30, 50, 75, 100])
    p.add_argument("--N_agents", type=int, default=2000)
    p.add_argument("--mu_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base_map_csv", default="base_map_potential_real_10m.csv")
    p.add_argument("--np_per_run", type=int, default=16)
    p.add_argument("--max_parallel", type=int, default=4,
                   help="Number of simultaneous MPI jobs. For 16 ranks/run, 4 uses 64 ranks.")
    p.add_argument("--mpirun", default=os.environ.get("MPIRUN", "mpirun"))
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.run_root)
    log_dir = root / "launcher_logs"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.max_parallel * args.np_per_run > 76:
        print(
            f"[warning] max_parallel*np_per_run = {args.max_parallel * args.np_per_run} > 76. "
            "A single SQUID CPU node has 76 cores. Use a larger allocation, reduce max_parallel, "
            "or submit each C_max as a separate job.",
            file=sys.stderr,
        )

    processes: list[tuple[float, str, subprocess.Popen, object, float]] = []
    records = []

    for cmax in args.cmax_values:
        run_id = f"N{args.N_agents}_C{fmt_param(cmax)}_mu{fmt_param(args.mu_scale)}_seed{args.seed}"
        out_dir = root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            args.mpirun, "-np", str(args.np_per_run),
            args.python, args.script,
            "--N_agents", str(args.N_agents),
            "--C_max", str(cmax),
            "--mu_scale", str(args.mu_scale),
            "--seed", str(args.seed),
            "--run_root", str(root),
            "--run_id", run_id,
            "--base_map_csv", args.base_map_csv,
        ]

        print("[launch]", " ".join(cmd))
        records.append({
            "run_id": run_id,
            "C_max": cmax,
            "N_agents": args.N_agents,
            "mu_scale": args.mu_scale,
            "seed": args.seed,
            "np_per_run": args.np_per_run,
            "command": " ".join(cmd),
            "output_dir": str(out_dir),
            "status": "DRY_RUN" if args.dry_run else "QUEUED",
        })
        if args.dry_run:
            continue

        while len(processes) >= args.max_parallel:
            still_running = []
            for pcmax, prid, proc, logf, t0 in processes:
                ret = proc.poll()
                if ret is None:
                    still_running.append((pcmax, prid, proc, logf, t0))
                else:
                    logf.close()
                    print(f"[done] {prid} C_max={pcmax} ret={ret} elapsed={time.time()-t0:.1f}s")
                    for r in records:
                        if r["run_id"] == prid:
                            r["status"] = "DONE" if ret == 0 else f"FAILED_{ret}"
                            r["walltime_sec"] = time.time() - t0
            processes = still_running
            if len(processes) >= args.max_parallel:
                time.sleep(30)

        log_path = log_dir / f"{run_id}.log"
        logf = open(log_path, "w", buffering=1)
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        processes.append((cmax, run_id, proc, logf, t0))
        for r in records:
            if r["run_id"] == run_id:
                r["status"] = "RUNNING"
                r["log_path"] = str(log_path)

    while processes:
        still_running = []
        for pcmax, prid, proc, logf, t0 in processes:
            ret = proc.poll()
            if ret is None:
                still_running.append((pcmax, prid, proc, logf, t0))
            else:
                logf.close()
                print(f"[done] {prid} C_max={pcmax} ret={ret} elapsed={time.time()-t0:.1f}s")
                for r in records:
                    if r["run_id"] == prid:
                        r["status"] = "DONE" if ret == 0 else f"FAILED_{ret}"
                        r["walltime_sec"] = time.time() - t0
        processes = still_running
        if processes:
            time.sleep(30)

    launcher_index = root / "launcher_index.csv"
    if records:
        keys = sorted(set().union(*(r.keys() for r in records)))
        with open(launcher_index, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
        print(f"[saved] {launcher_index}")


if __name__ == "__main__":
    main()
