"""
Generate a good quality figure of SO-101 with DH frames and lengths and frames

Robot rendered offscreen by MuJoCo and frames and dimensions are drawn on top with matplotlib, projected with the same camera,

Run:
    python tools/render_dh_figure.py #3/4 of the view
    python tools/render_dh_figure.py --side #Orthographic
    
Output:
    results/figures/dh_frames_render.png and .pdf
"""

import argparse
from pathlib import Path
import numpy as np
import mujoco
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parent.parent
XML=REPO/"model_files"/"so101_new_calib_frames.xml"
OUT=REPO/"results"/"figures"/"dh_frames_render"

W, H=2400, 1600
AXIS_LEN=0.035
ROBOT_RGBA=(0.78, 0.78, 0.80, 1.0)
MARGIN=70
LBL_BOX=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75)
MIN_SEG_PX=15
AX_COL={"x": "#D62728", "y": "#2CA02C", "z": "#1F77B4"}
D_COL="#1F4E9C"
A_COL="#C27C0E"

LABEL_OFFSET = {
    "persp": {"a1": (-42, 16), "d2": (-38, -16), "a3": (20, 24), "d4": (30, 30), "a5": (38, 16)},
    "side": {"a1": (-62, 0), "a5": (0, 22), "d5": (0, 16)},
}

def args_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--side", action="store_true", help="Orthographic side view")
    parser.add_argument("--q", type=float, nargs=5, default=[0, 0, 0, 0, 0], help="Joint Angles [rad]")
    return parser.parse_args()

def setup_mujoco(args):
    m=mujoco.MjModel.from_xml_path(str(XML))
    m.vis.global_.offwidth=W
    m.vis.global_.offheight=H
    m.geom_rgba[:]=ROBOT_RGBA
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.diffuse[:] = 0.55
    m.vis.headlight.specular[:] = 0.05
    d = mujoco.MjData(m)
    d.qpos[:5] = args.q
    mujoco.mj_forward(m, d)
    cam=mujoco.MjvCamera()
    cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:]=[0.21, 0.0, 0.16]
    if args.side:
        m.vis.global_.orthographic=1
        m.vis.global_.fovy=0.40
        cam.azimuth, cam.elevation, cam.distance=90, 0, 1.0
    else:
        cam.azimuth, cam.elevation, cam.distance=125, -22, 0.55
    opt=mujoco.MjvOption()
    opt.sitegroup[:]=0
    return m, d, cam, opt

def render(m, d, cam, opt):
    r=mujoco.Renderer(m, H, W)
    r.update_scene(d, cam, opt)
    rgb=r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(d, cam, opt)
    depth=r.render()
    r.disable_depth_rendering()
    robot=depth<depth.max()-1e-6
    rgb[~robot]=255
    return rgb, robot

def projector(m, cam):
    az=np.radians(cam.azimuth)
    el=np.radians(cam.elevation)
    fwd=np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    up=np.array([-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)])
    right=np.cross(fwd, up)
    pos=np.array(cam.lookat)-cam.distance*fwd
    ortho=bool(m.vis.global_.orthographic)
    fovy=m.vis.global_.fovy
    half_h=fovy/2 if ortho else np.tan(np.radians(fovy)/2)
    half_w=half_h*W/H
    
    def proj(p):
        v=np.asarray(p)-pos
        x, y=v@right, v@up
        if not ortho:
            z=v@fwd
            x,y=x/z, y/z
        return np.array([(x/half_w+1)/2*W, (1-y/half_h)/2*H])
    return proj

def dh_segments(m, d):
    sid=[m.site(f"dh{i}").id for i in range(6)]
    P=[d.site_xpos[s].copy() for s in sid]
    R=[d.site_xmat[s].reshape(3,3).copy() for s in  sid]
    segs=[]
    for i in range(1, 6):
        z=R[i-1][:, 2]
        di=float((P[i]-P[i-1])@z)
        mid=P[i-1]+di*z
        ai=float(np.linalg.norm(P[i]-mid))
        if abs(di)>1e-3:
            segs.append((P[i-1], mid, rf"$d_{i}={di*1000:.1f}$ mm", D_COL))
        if ai>1e-3:
            segs.append((mid, P[i], rf"$a_{i}={ai*1000:.1f}$ mm", A_COL))
    return P, R, segs

def main():
    args=args_parse()
    m, d, cam, opt=setup_mujoco(args)
    rgb, robot=render(m, d, cam, opt)
    proj=projector(m, cam)
    P, R, segs=dh_segments(m, d)
    
    pts=[proj(p+AXIS_LEN*1.3*Rm[:,k]) for p, Rm in zip(P, R) for k in range(3)]
    pts+=[proj(p) for seg in segs for p in seg[:2]]
    ys, xs=np.nonzero(robot)
    pts=np.vstack([pts, np.c_[xs[[xs.argmin(), xs.argmax()]], ys[[ys.argmin(), ys.argmax()]]]])
    u0, v0=np.maximum(pts.min(0)-MARGIN, 0).astype(int)
    u1, v1=np.minimum(pts.max(0)+MARGIN, [W, H]).astype(int)
    rgb=rgb[v0:v1, u0:u1]
    off=np.array([u0, v0])
    P2=lambda p: proj(p)-off
    centre=np.mean([P2(p) for p in P], axis=0)
    
    h, w=rgb.shape[:2]
    fig=plt.figure(figsize=(w/300, h/300), dpi=300)
    ax=fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb)
    ax.set_axis_off()
    
    overrides=LABEL_OFFSET["side" if args.side else "persp"]
    hidden=[]
    for a, b, label, col in segs:
        pa=P2(a)
        pb=P2(b)
        if np.linalg.norm(pb-pa)<MIN_SEG_PX:
            hidden.append(label)
            continue
        ax.plot(*zip(pa, pb), color=col, lw=1.4, ls=(0, (4, 2)), zorder=3)
        mid=(pa+pb)/2
        t=pb-pa
        n=np.array([-t[1], t[0]])/(np.linalg.norm(t)+1e-9)
        if n@(mid-centre)<0:
            n=-n
        key=label[1]+label[3]
        xy=overrides.get(key, tuple(18*n*[1, -1]))
        ax.annotate(label, mid, xytext=xy, textcoords="offset points", fontsize=6.5, color=col, ha="center", va="center", zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, lw=0.5))
    if hidden:
        ax.text(0.99, 0.02, "Perpendicular to the view: "+", ".join(hidden), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=D_COL, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=D_COL, lw=0.5))
    labels={}
    for i, (p, Rm) in enumerate(zip(P, R)):
        o=P2(p)
        ax.plot(*o, "ko", ms=2.5, zorder=6)
        for k, name in enumerate("xyz"):
            e=P2(p+AXIS_LEN*Rm[:,k])
            if np.linalg.norm(e-o)<10:
                continue
            ax.annotate("", e, o, zorder=6, arrowprops=dict(arrowstyle="-|>", color=AX_COL[name], lw=1.2, mutation_scale=7))
            pos=o+1.25*(e-o)
            key=tuple(np.round(pos/20).astype(int))
            labels.setdefault(key, [pos, []])[1].append((rf"{name}_{i}", AX_COL[name]))
    for pos, items in labels.values():
        if len(items)==1:
            txt, col=items[0]
            ax.text(*pos, f"${txt}$", color=col, fontsize=6.5, ha="center", va="center", zorder=7, bbox=LBL_BOX)
        else:
            ax.text(*pos, "$"+",\\,".join(t for t,_ in items)+"$", color="#444444", fontsize=6.5, ha="center", va="center", zorder=7, bbox=LBL_BOX)
    
    OUT.parent.mkdir(parents=True, exist_ok=True)
    suffix="_side" if args.side else ""
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}{suffix}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    print(f"Saved {OUT}{suffix}.png and .pdf")
    
if __name__=="__main__":
    main()        