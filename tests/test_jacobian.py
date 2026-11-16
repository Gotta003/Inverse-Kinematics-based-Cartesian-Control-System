import sys
from pathlib import Path

REPO_ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

def main():
    np.set_printoptions(precision=4, suppress=True)

if __name__=="__main__":
    main()