"""
fk_hardware.py
Forward-Kinematics check on physical SO-101

For each pose the script will print the TCP positions, to enable measures and same those in results/data/fk_hardware.csv

Measures:
    x_rel: Forward
    y_rel: Left
    z_rel: Height above table
    
Modes:
    commanded (default): The robot moves, with torque ON, to a list of
    predefined joint poses (checked in MuJoCo: no collisions, TCP above the
    table)
    passive: Torque OFF. You move the arm by hand and press Enter to
    read it.
    
Usage:
-------
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))