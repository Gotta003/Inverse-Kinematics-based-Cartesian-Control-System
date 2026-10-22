import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np



if __name__=="__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("Screw Axes:")
    print("M")
    print("T_0E")
    print("Links Lengths [m]:")
    