import sys
import time
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import numpy as np

from src.utils.constants import ALL_JOINT_NAMES, JOINT_NAMES, LOOP_HZ, POSES
from utils.so101_utils import setup_motors, load_calibration

class Arm:
    def __init__(self, bus, degrees_mode):
        self.bus=bus
        self.degrees_mode=degrees_mode
        
    @classmethod
    def connect(cls, robot_name, port):
        calibration=load_calibration(robot_name)
        bus=setup_motors(calibration, port)
        return cls(bus, degrees_mode=False)
    
    def close(self):
        self.bus.disconnect()
        
    def disable_torque(self):
        self.bus.disable_torque()
        
    def _to_model(self, raw):
        out=dict(raw)
        if self.degrees_mode:
            for j in JOINT_NAMES:
                if j in out:
                    out[j]=np.radians(out[j])
        return out
    
    def _to_bus(self, cmd):
        out=dict(cmd)
        if self.degrees_mode:
            for j in JOINT_NAMES:
                if j in out:
                    out[j]=float(np.degrees(out[j]))
        return out
    
    def read(self):
        return {
            j: float(v) for j, v in self._to_model(self.bus.sync_read("Present_Position")).items()
        }
        
    def read_arm(self):
        q=self.read()
        return np.array([q[j] for j in JOINT_NAMES])
    
    def write(self, cmd):
        self.bus.sync_write("Goal_Position", self._to_bus(cmd), normalize=True)
        
    def freeze(self):
        self.write(self.read())
        
    def move(self, target, duration, log=None, t_offset=0.0):
        start=self.read()
        t0=time.perf_counter()
        while True:
            t=time.perf_counter()-t0
            s=min(t/duration, 1.0) if duration>0 else 1.0
            cmd={j: (1-s)*start[j]+s*target[j] for j in target}
            self.write(cmd)
            if log is not None:
                log.append((t_offset+t, {**start, **cmd}, self.read()))
            if s>=1.0:
                break
            time.sleep(max(0.0, 1.0/LOOP_HZ-(time.perf_counter()-t0-t)))
        return time.perf_counter()-t0
    
    def hold(self, duration, log=None, t_offset=0.0):
        t0=time.perf_counter()
        while(t:=time.perf_counter()-t0)<duration:
            if log is not None:
                log.append((t_offset+t, None, self.read()))
            time.sleep(1.0/LOOP_HZ)
        return time.perf_counter()-t0