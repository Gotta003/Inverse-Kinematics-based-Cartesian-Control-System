# Inverse-Kinematics-based-Cartesian-Control-System
The primary objective is to develop an inverse-kinematics-based Cartesian control system that enables the robot end-effector to move to desired positions or poses. The project integrates: forward-kineamtics, inverse kinematics, manipulator Jacobians, optimization, joint-limit, singularity awareness, trajectory, control, simulation and hardware. 

## Phase 1 - FK: How to Run

**First time**: create the `lerobot` environment and install the dependencies:
```bash
chmod +x setup.sh run.sh
./setup.sh
```
The setup is under test. If this run didn't work out as expected, you can install the missing packages present in `requirements.txt` manually.

**Every time**: activate the environment and start the launcher from the repo root:
```bash
conda activate lerobot
./run.sh
```

`run.sh` opens a terminal menu implemented in `run.py`. Type the number of an option and press enter following the instructions on console, some require extra inputs. To maintain the default setups many times is sufficient to press Enter.

| # | Option | What it does | Output |
|---|--------|--------------|--------|
| 1 | FK tests (pytest) | Checks `SO_101_fk` against MuJoCo on the named poses | terminal |
| 2 | FK validation vs MuJoCo | Compares the PoE FK with MuJoCo on random configurations (number and seed are asked); optionally steps through the poses in the viewer | `results/data/fk_validation.csv` |
| 3 | FK check on the real arm | *commanded*: the arm moves to the safe poses; *passive*: torque off, move the arm by hand (hold it!) | `results/data/fk_hardware.csv` |
| 4 | Simulation vs real arm | Runs the same trajectory in MuJoCo and on the robot and compares the joints | `results/data/sim_vs_real.csv`, `results/figures/sim_vs_real_*.png` |
| 5 | Check robot connection | Reads the joints and checks they are in radians | terminal |
| 6 | Detect the port again | Finds the robot's serial port (unplug/replug procedure) | `.robot_config.json` |
| 7 | Change robot | Selects the calibration file in `calibration_files/` | `.robot_config.json` |
| 8 | Extract PoE | Prints the screw axes and home pose from the MuJoCo model, to copy into `src/utils/constants.py` | terminal |
| 9 | Render DH frames figure | Renders the SO-101 with its DH frames and lengths (3/4 or side view, any `q`) | `results/figures/dh_frames_render*.png/.pdf` |

Options 3–5 need the arm connected via USB. The first time, the launcher asks you to detect the port and choose the robot, and it saves both in `.robot_config.json` for the next runs. Options 1, 2, 8 and 9 only use the simulation