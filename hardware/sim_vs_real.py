"""
sim_vs_real.py
Performs the joint trajectory in MuJoCo first, then on real arm and compare, the output is a live figure graph stack time/angle per joint + gripper

Flow:
1) Read real arm's current pose
2) Simulate start->waypoints->start in MuJoCo
3) If simulation clean, type go to run the same on arm
4) Compare the two partnership

Outputs:
    results/data/sim_vs_real.py
    results/figures/sim_vs_real_joints.png  commanded in MuJoCo and Real
    results/figures/sim_vs_real_tcp.png     TCP path and Cartesian error

Usage:
    python hardware/sim_vs_real.py --port /dev/ttyACM0
    python hardware/sim_vs_real.py --port /dev/ttyACM0 --poses zero reach_forward compact
    python hardware/sim_vs_real.py --port /dev/ttyACM0 --no-viewer
"""
import sys
from pathlib import Path
REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import argparse
import csv
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from src.arm import Arm
from src.forward_kinematics import SO_101_fk
from src.utils.constants import (
    ALL_JOINT_NAMES,
    JOINT_NAMES,
    LOOP_HZ,
    MOVE_TIME,
    PLOT_PERIOD,
    POSES,
    SETTLE_TIME,
)
from src.utils.helper import interp

SCENE_XML=REPO_ROOT/"model_files"/"scene.xml"
DATA_DIR=REPO_ROOT/"results"/"data"
FIG_DIR=REPO_ROOT/"results"/"figures"

DEFAULT_SEQUENCE=["zero", "elbow_up", "high_reach", "reach_forward", "wrist_down", "compact", "side_mixed"]

def gripper_to_sim(g):
    return g*np.pi/100.0

def gripper_from_sim(r):
    return r*100.0/np.pi

def build_plan(start, waypoints, dt=1.0/LOOP_HZ):
    ts, cmds, t, prev=[], [], 0.0, dict(start)
    for wp in waypoints:
        for duration, moving in ((MOVE_TIME, True), (SETTLE_TIME, False)):
            n=int(duration/dt)
            for k in range(n):
                cmds.append(interp(prev, wp, k/n, ALL_JOINT_NAMES) if moving else wp)
                ts.append(t)
                t+=dt
        prev=wp
    return np.array(ts), {j: np.array([c[j] for c in cmds]) for j in ALL_JOINT_NAMES}, t

class LivePlot:
    def __init__(self, plan_t, plan_cmd, total_time):
        plt.ion()
        n=len(ALL_JOINT_NAMES)
        self.fig, self.axes=plt.subplots(n, 1, figsize=(10, 1.9*n), sharex=True)
        self.data={s:{
            "t": [],
            **{j: [] for j in ALL_JOINT_NAMES}
        } for s in ("sim", "real")}
        self.lines={"sim": {}, "real": {}}
        for ax, j in zip(self.axes, ALL_JOINT_NAMES):
            lo, hi=plan_cmd[j].min(), plan_cmd[j].max()
            pad=max(0.15 if j!="gripper" else 10.0, 0.25*(hi-lo))
            ax.plot(plan_t, plan_cmd[j], "k--", lw=1, label="commanded")
            self.lines["sim"][j],=ax.plot([], [], color="tab:blue", lw=2.5, alpha=0.55, label="MuJoCo", animated=True)
            self.lines["real"][j],=ax.plot([], [], color="tab:orange", lw=1.5, label="real", animated=True)
            ax.set_ylim(lo-pad, hi+pad)
            ax.set_ylabel(f"{j}\n[{'0-100' if j=='gripper' else 'rad'}]")
            ax.grid(alpha=0.3)
        self.axes[0].set_xlim(0, total_time*1.05)
        self.axes[0].legend(loc="upper right", fontsize=8, ncol=3)
        self.axes[-1].set_xlabel("t [s]")
        self.title=self.fig.suptitle("Simulation running...", animated=True)
        self.fig.tight_layout()
        self.bg=None
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self._last=0.0
        
    def _on_draw(self, _event):
        self.bg=self.fig.canvas.copy_from_bbox(self.fig.bbox)
        for a in self._artists():
            self.fig.draw_artist(a)
            
    def add(self, source, t, q):
        d=self.data[source]
        d["t"].append(t)
        for j in ALL_JOINT_NAMES:
            d[j].append(q[j])
            
    def refresh(self, force=False):
        now=time.perf_counter()
        if not force and now-self._last<PLOT_PERIOD:
            return
        self._last=now
        for source, lines in self.lines.items():
            d=self.data[source]
            for j, ln in lines.items():
                ln.set_data(d["t"], d[j])
        canvas=self.fig.canvas
        if self.bg is None:
            canvas.draw()
        canvas.restore_region(self.bg)
        for a in self._artists():
            self.fig.draw_artist(a)
        canvas.blit(self.fig.bbox)
        canvas.flush_events()
    
    def set_status(self, text):
        self.title.set_text(text)
        self.refresh(force=True)
    
    def _artists(self):
        return [self.title] + [ln for src in self.lines.values() for ln in src.values()]
    
    def save(self, path):
        for a in self._artists():
            a.set_animated(False)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=120)

class LiveLog(list):
    def __init__(self, plot, source):
        super().__init__()
        self.plot=plot
        self.source=source
        
    def append(self, row):
        super().append(row)
        self.plot.add(self.source, row[0], row[2])
        self.plot.refresh()

def run_simulation(start, waypoints, live, arm, show_viewer=True):
    m=mujoco.MjModel.from_xml_path(str(SCENE_XML))
    d=mujoco.MjData(m)
    floor={i for i in range(m.ngeom) if m.geom(i).name=="floor"}
    
    def to_sim(q):
        return [q[j] for j in JOINT_NAMES]+[gripper_to_sim(q["gripper"])]
    
    def from_sim():
        q={
            j: float(d.qpos[i]) for i, j in enumerate(JOINT_NAMES)
        }
        q["gripper"]=gripper_from_sim(float(d.qpos[5]))
        return q
    
    d.qpos[:6]=to_sim(start)
    d.ctrl[:]=to_sim(start)
    mujoco.mj_forward(m, d)
    
    def current_contacts():
        out=set()
        for c in d.contact[:d.ncon]:
            b1=m.body(m.geom_bodyid[c.geom1]).name
            b2=m.body(m.geom_bodyid[c.geom2]).name
            kind="floor" if (c.geom1 in floor or c.geom2 in floor) else "self"
            out.add((kind, *sorted((b1, b2))))
        return out
    
    baseline=current_contacts()
    if baseline:
        print(f"Start pose already in contact (allowed when leaving/returning): {sorted(baseline)}")
    last_seg=len(waypoints)-1
    
    viewer=None
    if show_viewer:
        viewer=mujoco.viewer.launch_passive(m, d)
    
    sim_log, contacts=LiveLog(live, "sim"), set()
    dt=m.opt.timestep
    every=max(1, round(1.0/LOOP_HZ/dt))
    t_sim=0.0
    wall0=time.perf_counter()
    prev=dict(start)
    for seg, wp in enumerate(waypoints):
        near_start=(seg==0) or (seg==last_seg)
        for duration, moving in ((MOVE_TIME, True), (SETTLE_TIME, False)):
            n=int(duration/dt)
            for k in range(n):
                cmd=interp(prev, wp, k/n, ALL_JOINT_NAMES) if moving else wp
                d.ctrl[:]=to_sim(cmd)
                mujoco.mj_step(m, d)
                for c in current_contacts():
                    if near_start and c in baseline:
                        continue
                    contacts.add(c)
                    b1=m.body(m.geom_bodyid[c.geom1]).name
                    b2=m.body(m.geom_bodyid[c.geom2]).name
                    kind="floor" if (c.geom1 in floor or c.geom2 in floor) else "self"
                    contacts.add((kind, *sorted((b1, b2))))
                if k%every==0:
                    sim_log.append((t_sim, cmd, from_sim()))
                t_sim+=dt
                if viewer is not None:
                    if viewer.is_running():
                        viewer.sync()
                    else:
                        viewer=None
                lag=t_sim-(time.perf_counter()-wall0)
                if lag>0:
                    time.sleep(lag)
        prev=wp
    if viewer is not None:
        viewer.close()
    live.refresh(force=True)
    floor_hits=sorted(c for c in contacts if c[0]=="floor")
    self_hits=sorted(c for c in contacts if c[0]=="self")
    if floor_hits or self_hits:
        msg=f"Simulation: CONTACTS {floor_hits+self_hits}"
        print(msg+"\nHardware run blocked: change sequence")
        live.set_status(msg)
        live.save(FIG_DIR/"sim_only.png")
        arm.close()
        return
    live.set_status("Simulation clean: type 'go' in terminal to run the real arm")
    print("Simulation finished, no contacts. Check the plots before continuing.")
    if input("Run same trajectory on real arm? Type 'go': ").strip()!="go":
        live.save(FIG_DIR/"sim_only.png")
        input("HOLD ARM, then press Enter to turn torque OFF")
        arm.close()
        return
    return sim_log

def run_hardware(waypoints, live, arm):
    real_log, t=LiveLog(live, "real"), 0.0
    live.set_status("Hardware running... (Ctrl+C freezes arm)")
    try:
        for i, wp in enumerate(waypoints):
            print(f"->waypoint {i+1}/{len(waypoints)}")
            t+=arm.move(wp, MOVE_TIME, log=real_log, t_offset=t)
            t+=arm.hold(SETTLE_TIME, log=real_log, t_offset=t)
        live.set_status("Hardware run finished")
    except KeyboardInterrupt:
        arm.freeze()
        print("\nInterrupted: arm frozen in place")
        live.set_status("Interrupted: arm frozen in place")
    finally:
        input("HOLD ARM, then Enter to release torque and disconnect")
        arm.close()
    return real_log

def unpack(log):
    ts=np.array([r[0] for r in log])
    cmd={j: [] for j in ALL_JOINT_NAMES}
    measures={j: [] for j in ALL_JOINT_NAMES}
    last=log[0][2]
    for _, c, q in log:
        last=c if c is not None else last
        for j in ALL_JOINT_NAMES:
            cmd[j].append(last[j])
            measures[j].append(q[j])
    return ts, {j: np.array(v) for j, v in cmd.items()}, {j: np.array(v) for j, v in measures.items()}

def real_to_sim_time(real_log):
    seg_len=MOVE_TIME+SETTLE_TIME
    segs, prev_hold=[], True
    for t, cmd, _ in real_log:
        is_move=cmd is not None
        if is_move and prev_hold:
            segs.append({"move": [], "hold": []})
        segs[-1]["move" if is_move else "hold"].append(t)
        prev_hold=not is_move
    mapped=[]
    for k, s in enumerate(segs):
        for part, dur, offset in (("move", MOVE_TIME, 0.0), ("hold", SETTLE_TIME, MOVE_TIME)):
            ts=s[part]
            if not ts:
                continue
            span=max(ts[-1]-ts[0], 1e-9)
            mapped+=[k*seg_len+offset+(t-ts[0])/span*dur for t in ts]
    return np.array(mapped)

def compare(sim_log, real_log, live):
    ts_s, _, sim_q=unpack(sim_log)
    ts_r, real_cmd, real_q=unpack(real_log)
    t_map=real_to_sim_time(real_log)
    sim_at_r={j: np.interp(t_map, ts_s, sim_q[j]) for j in ALL_JOINT_NAMES}
    
    seg_len=MOVE_TIME+SETTLE_TIME
    n_seg=int(t_map.max()//seg_len)+1
    core=t_map<(n_seg-1)*seg_len
    if not core.any():
        core=np.ones_like(t_map, dtype=bool)
    
    def tcp(qd, k):
        return SO_101_fk(np.array([qd[j][k] for j in JOINT_NAMES]))[:3, 3]
    
    p_real=np.array([tcp(real_q, k) for k in range(len(ts_r))])
    p_sim=np.array([tcp(sim_at_r, k) for k in range(len(ts_r))])
    p_cmd=np.array([tcp(real_cmd, k) for k in range(len(ts_r))])
    err_sim=np.linalg.norm(p_real-p_sim, axis=1)*1000
    err_cmd=np.linalg.norm(p_real-p_cmd, axis=1)*1000
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR/"sim_vs_real.csv", "w", newline="") as f:
        w=csv.writer(f)
        w.writerow(
            ["t"] + 
            [f"cmd_{j}" for j in ALL_JOINT_NAMES] + 
            [f"real_{j}" for j in ALL_JOINT_NAMES] +
            [f"sim_{j}" for j in ALL_JOINT_NAMES] + [
                "tcp_real_x", 
                "tcp_real_y", 
                "tcp_real_z", 
                "tcp_sim_x", 
                "tcp_sim_y", 
                "tcp_sim_z",
                "tcp_err_real_sim_mm", 
                "tcp_err_real_cmd_mm"
        ])
        for k, tk in enumerate(ts_r):
            w.writerow(
                [tk] +
                [real_cmd[j][k] for j in ALL_JOINT_NAMES] +
                [real_q[j][k] for j in ALL_JOINT_NAMES] +
                [sim_at_r[j][k] for j in ALL_JOINT_NAMES] +
                list(p_real[k]) +
                list(p_sim[k]) +
                [err_sim[k], err_cmd[k]]
            )
    print("\nRMS error:              real vs command   real vs sim")
    for ax, j in zip(live.axes, ALL_JOINT_NAMES):
        rms_cmd=np.sqrt(np.mean((real_q[j]-real_cmd[j])**2))
        rms_sim=np.sqrt(np.mean((real_q[j]-sim_at_r[j])**2))
        unit="" if j=="gripper" else " rad"
        print(f"  {j:<14}       {rms_cmd:10.4f}{unit:4s}   {rms_sim:10.4f}{unit}")
        ax.text(0.01, 0.92, f"RMS real-cmd {rms_cmd:.3f}    real-sim {rms_sim:.3f}{unit}", transform=ax.transAxes, fontsize=8, va="top", bbox=dict(fc="white", ec="none", alpha=0.7))
    print(f"TCP (FK) real vs sim    : max {err_sim.max():.1f} mm, RMS {np.sqrt(np.mean(err_sim**2)):.1f} mm")
    print(f"TCP (FK) real vs command: max {err_cmd.max():.1f} mm, RMS {np.sqrt(np.mean(err_cmd**2)):.1f} mm")
    
    fig, axes=plt.subplots(1, 3, figsize=(15, 4.5))
    for p, lab, sty in ((p_cmd, "commanded", "k--"), (p_sim, "MuJoCo", "-"), (p_real, "real", "-")):
        axes[0].plot(p[:,0], p[:,2], sty, label=lab)
        axes[1].plot(p[:,0], p[:,1], sty, label=lab)
    axes[0].set(xlabel="x [m]", ylabel="z [m]", title="TCP path, side view")
    axes[1].set(xlabel="x [m]", ylabel="y [m]", title="TCP path, top view")
    for ax in axes[:2]:
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].plot(ts_r, err_sim, label="real vs MuJoCo")
    axes[2].plot(ts_r, err_cmd, label="real vs commanded")
    if (~core).any():
        axes[2].axvspan(ts_r[~core][0], ts_r[-1], color="0.85", zorder=0, label="return to rest (excluded)")
    axes[2].set(xlabel="t [s]", ylabel="TCP error [mm]", title="Cartesian error (via FK)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR/"sim_vs_real_tcp.png", dpi=150)

def args_parse():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True, help="Serial port, like /dev/ttyACM0")
    parser.add_argument("--robot", default="jarvis", help="calibration_files/<robot>.json")
    parser.add_argument("--poses", nargs="+", default=DEFAULT_SEQUENCE, choices=list(POSES))
    parser.add_argument("--no-viewer", action="store_true", help="no MuJoCo 3D viewer")
    return parser.parse_args()

def main():
    args=args_parse()
    arm=Arm.connect(args.robot, args.port)
    start=arm.read()
    gripper=start["gripper"]
    waypoints=[{**dict(zip(JOINT_NAMES, POSES[n])), "gripper": gripper} for n in args.poses]
    waypoints.append(dict(start))
    print(f"Sequence: start -> {' -> '.join(args.poses)} -> start")
    plan_t, plan_cmd, total=build_plan(start, waypoints)
    live=LivePlot(plan_t, plan_cmd, total)
    sim_log=run_simulation(start, waypoints, live, arm, show_viewer=not args.no_viewer)
    if sim_log is None:
        return
    real_log=run_hardware(waypoints, live, arm)
    if len(real_log)<2:
        print("No hardware data recorded")
        return
    compare(sim_log, real_log, live)
    live.save(FIG_DIR/"sim_vs_real_joints.png")
    print(f"Saved {DATA_DIR/'sim_vs_real.csv'} and figures in {FIG_DIR}")
    plt.ioff()
    plt.show()
if __name__=="__main__":
    main()