import time
import numpy as np
import mujoco
import mujoco.viewer

XML="model_files/SO101_new_calib_frames.xml"
m=mujoco.MjModel.from_xml_path(XML)
d=mujoco.MjData(m)
sid=[m.site(f"dh{i}").id for i in range(6)]
D_COL, A_COL = np.array([0.2, 0.6, 1, 1]), np.array([1, 0.8, 0.1, 1])

def segments():
    segs=[]
    for i in range(1,6):
        p0, z0=d.site_xpos[sid[i-1]], d.site_xmat[sid[i-1]].reshape(3,3)[:,2]
        p1=d.site_xpos[sid[i]]
        di=float(np.dot(p1-p0, z0))
        mid=p0+di*z0
        ai=float(np.linalg.norm(p1-mid))
        if abs(di)>1e-3:
            segs.append((p0, mid, f"d{i}={abs(di)*1000:.1f} mm", D_COL))
        if ai>1e-3:
            segs.append((mid, p1, f"a{i}={ai*1000:.1f} mm", A_COL))
    return segs

def draw(scn):
    scn.ngeom=0
    for a, b, label, rgba in segments():
        g=scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba.astype(np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.0015, a, b)
        g.label=label
        scn.ngeom+=1
        
if __name__=="__main__":
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.lookat[:]=[0.20, 0.0, 0.15]
        v.cam.distance=0.75
        v.cam.azimuth=90
        v.cam.elevation=-5
        v.opt.frame=mujoco.mjtFrame.mjFRAME_SITE
        v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]=True
        while v.is_running():
            t=time.time()
            mujoco.mj_step(m, d)
            with v.lock():
                draw(v.user_scn)
            v.sync()
            time.sleep(max(0, m.opt.timestep-(time.time()-t)))
