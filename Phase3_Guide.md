# Phase 3 - Cartesian Control and Final Demonstration

The final phase integrates the IK of Phase 2 with robot control: the TCP has to move along smooth Cartesian trajectories between commanded positions, tracing a predefined geometric shape (Option A) or a sequence of Cartesian waypoints (Option B), in MuJoCo and on the real arm. The motion must show that the controller works in Cartesian space, not that it replays joint configurations. Scope of this guide: trajectory generation, shapes, robot control, tests, simulation, hardware and report (the finger/camera teleoperation is not part of it). The submission must include:
- [ ] Cartesian command interface (`src/robot_control.py`): a target `target_position=[x, y, z]` or `target_pose=T_desired` is accepted (In Report -> Controller: Cartesian commands)
- [ ] Controller loop (`src/robot_control.py`) with the 4 steps required: 1) IK of the target, 2) trajectory from current to desired, 3) command the simulated or physical robot, 4) decide whether the target is reached (In Report -> Controller implementation and parameters)
- [ ] Smooth trajectories between targets/waypoints (`src/trajectory.py`) with quintic time scaling and speed/acceleration limits (In Report -> Trajectory generation)
- [ ] Option A: predefined shapes (`src/trajectory.py`), at least square, rectangle, triangle, circle (In Report -> Demonstration task)
- [ ] Option B: waypoint sequence P1 -> P2 -> ... -> PN (`src/trajectory.py`) (same section, optional but cheap once Option A works)
- [ ] Tests of the trajectory and of the controller (`tests/test_trajectory.py`) (In Report -> Results: simulation)
- [ ] Validation in MuJoCo of every shape (`simulation/trajectory_sim.py`) with tracking/path errors, CSVs and figures (In Report -> Results: table and figure)
- [ ] Safe demonstration on the real arm (`hardware/trajectory_demo.py`) with dry run, checks before every move and measured errors (In Report -> Results: hardware)
- [ ] README updated with how to run the simulation and the final demonstration, and the main files (Deliverable 1)
- [ ] Video or live demo of the physical SO-101 tracing the shape (Deliverable 3)

## Controller structure
The controller works in three layers, everything is planned and checked BEFORE the arm moves:
1) **Path (where)**: the TCP path in Cartesian space, built from pieces:
    - `Line` (polygons, waypoints), `Arc` (rounded corners if wanted), `Curve` (parametric: circle, ellipse, figure 8, ...)
    - drawn on a plane: `horizontal` (parallel to the table, at height z) or `vertical` (in front of the arm, facing it)
2) **Time law (how)**: on every piece a rest-to-rest quintic $s(\tau)=10\tau^3-15\tau^4+6\tau^5$, $\tau=t/T$
    - position, velocity and acceleration continuous; velocity and acceleration zero at both ends, so the arm stops at every corner/waypoint (no jerk spikes, and the "reached" check is made at rest)
    - duration $T$ = the shortest one respecting the TCP limits: $v_{peak}=\frac{15}{8}\frac{L}{T}\le v_{max}$, $a_{peak}=\frac{10}{\sqrt3}\frac{L}{T^2}\le a_{max}$, $T\ge T_{min}$
    - on curves also the centripetal part: $v(t)^2\kappa(t)\le a_{max}/2$ (slower where it bends)
3) **Joints**: the Cartesian trajectory is sampled every $1/$`LOOP_HZ` s and every sample is solved with the Phase 2 IK (pitch mode, tool pitch held constant, `wrist_roll` held), warm started from the previous sample with no restarts, so it stays on the same solution branch
    - if a joint would move faster than `TRAJ_JOINT_V_MAX`, all durations are stretched and the path is replanned (retiming)
    - start pose -> first point of the path and last point -> start: joint-space quintic (no Cartesian requirement there, only smooth and collision free)

Session on the arm: start (rest) pose -> HOME (0, 0, -90, 0, 0) deg -> choose a motion -> HOME -> first point -> shape -> HOME -> ... -> quit: HOME -> start pose -> torque off. Start <-> HOME transitions use the same relaxed rules as `hardware/ik_hardware.py` (contacts and height already present at the rest pose are allowed, never lower, no new contacts); everything else uses the strict checks.

## Files
| Path | Content | Status | Role |
|---|---|---|---|
| `src/forward_kinematics.py`, `src/jacobian.py`, `src/inverse_kinematics.py` | FK, Jacobian, IK | from Phases 1-2 | DONE |
| `src/trajectory.py` | path pieces, shapes, waypoints, quintic time law, Cartesian -> joint planning, retiming | draft from earlier version, to port | |
| `src/robot_control.py` | Cartesian command API, plan checks, executor (sim/real), reached check, logging | new (split out of the draft `trajectory_demo.py`) | |
| `src/utils/constants.py` | add a trajectory block (`TRAJ_*`, `SHAPE_*`, `HOME_POSE_DEG`) | extended | |
| `src/utils/sim_checks.py` | add a path check (contacts and lowest TCP along a joint trajectory) | extended | |
| `tests/test_trajectory.py` | time law, shapes, limits, IK tracking, controller checks | draft, to port | |
| `simulation/trajectory_sim.py` | all shapes in MuJoCo, table of errors, CSVs, figures, viewer | new | |
| `hardware/trajectory_demo.py` | final demonstration on the real arm (session, dry run, checks, measures) | draft, to port | |
| `run.py`, `README.md` | Phase 3 menu entries and instructions | extended | |
| `results/data/`, `results/figures/` | `trajectory_<shape>_<sim\|hw>.csv/.png`, summary table | generated | |

## Porting the earlier draft to the Phase 2 IK
The earlier `src/trajectory.py` and `hardware/trajectory_demo.py` were written against an older IK and do not import anymore:
- [ ] `from src.inverse_kinematics import step, task, clip_norm` -> these do not exist; replace the per-sample `ik_track()` with `inverse_kinematics(p, q0=q_prev, mode="pitch", pitch=pitch, limits=lim, restarts=0, max_iterations=TRAJ_IK_ITERS)` (warm start, no restarts)
- [ ] Per-sample tolerance: the solver stops at `IK_POS_TOL`=0.1 mm; for a path a tighter one (about 1e-6 m) avoids small jumps between samples, so add an optional `pos_tol` argument to `inverse_kinematics` (default `IK_POS_TOL`, nothing else changes)
- [ ] `IK_MAX_POS_STEP`, `IK_MAX_ROT_STEP` -> now `IK_MAX_POS_ERR`, `IK_MAX_ORI_ERR`
- [ ] `ik_limits` -> `inv_kin_limits` (`src/utils/limits.py`)
- [ ] `check_path` is missing in `src/utils/sim_checks.py` -> add it (see below)
- [ ] Add the trajectory block to `constants.py`
- [ ] Run `tests/test_trajectory.py` and fix until green

## `src/trajectory.py`
- [ ] Time law:
    - [ ] `quintic(tau)` returning $s$, $ds/d\tau$, $d^2s/d\tau^2$
    - [ ] `quintic_duration(L, v_max, a_max, t_min)` shortest duration within the limits
- [ ] Path pieces with a common interface (`length`, `point(s)`, `tangent(s)`, `curvature(s)`): `Line`, `Arc`, `Curve`
- [ ] `shape_path(kind, centre, size, z, plane, ...)` for Option A:
    - [ ] required: `square`, `rectangle`, `triangle`, `circle`
    - [ ] extra (optional): `polygon` (n sides), `star`, `ellipse`, `figure8`, `heart`, `spiral`, `helix`
    - [ ] `horizontal` and `vertical` planes
- [ ] `polyline(points)` for Option B (validates at least 2 points, no duplicates)
- [ ] `CartesianTrajectory`: segments with durations from the limits (plus centripetal check on curves), dwell at every corner, `sample(t)` returning position, velocity, acceleration and the segment index
- [ ] `plan_joint_trajectory(traj, q_first, pitch, ...)`: IK of every sample, warm started, returns a `JointPlan` (t, q, desired p, IK errors, segment index); retime if joints too fast, give up after a few retimings
- [ ] `joint_quintic(q0, q1)` for approach/return and `full_plan(q_start, q_first, path_plan, q_end)` that chains them
- [ ] `reached(p_tcp, p_target, tol)`
- [ ] `__main__` that plans a square and prints durations, max speeds and IK errors

## `src/robot_control.py`
- [ ] `move_to(target_position | target_pose)`: the Cartesian command of the brief, IK -> joint quintic from the current pose -> checks -> execute -> reached
- [ ] `run_path(traj)`: approach, path, return, with the same steps
- [ ] `check_plan(...)` before every move, each check printed PASS/FAIL with the reason (same convention as `ik_hardware.py`):
    - [ ] IK error of every sample below a threshold (e.g. 1 mm)
    - [ ] every sample inside the IK joint limits
    - [ ] joint speed below `TRAJ_JOINT_V_MAX`
    - [ ] no new contacts in MuJoCo along the whole plan
    - [ ] TCP above the table (20-30 mm) along the path
    - [ ] start <-> HOME relaxed rules only for those two transitions
- [ ] `Executor` with the same interface for MuJoCo and for `Arm`: send the planned q at `LOOP_HZ`, read back q, log (t, q_cmd, q_read)
- [ ] Ctrl+C freezes the arm (`arm.freeze()`), torque released only after "HOLD ARM"
- [ ] After a motion: reached flag at every corner/waypoint (TCP after the dwell within `--tol`), tracking error (TCP vs desired at the same time), path error (TCP vs closest point of the path)

## `src/utils/`
- [ ] `constants.py` trajectory block:
    - [ ] `TRAJ_V_MAX` (≈0.04 m/s, conservative), `TRAJ_A_MAX` (≈0.2 m/s²), `TRAJ_T_MIN`, `TRAJ_DWELL`
    - [ ] `TRAJ_JOINT_V_MAX`, `TRAJ_JOINT_A_MAX`, `TRAJ_APPROACH_T_MIN`
    - [ ] `TRAJ_IK_TOL`, `TRAJ_IK_ITERS`
    - [ ] `SHAPE_CENTRE`, `SHAPE_SIZE`, `SHAPE_Z`, `HOME_POSE_DEG`
- [ ] `sim_checks.py`: `check_path(m, d, Q, allowed=set())` returning new contacts and the lowest TCP along a list of configurations

## `tests/test_trajectory.py`
- [ ] Time law: boundary conditions, derivatives vs finite differences, peaks $15/8$ and $10/\sqrt3$, durations respect the limits and the minimum
- [ ] Pieces: line and arc lengths, points, tangents; curves vs analytic lengths
- [ ] Shapes: every shape closed (except spiral/helix), in its plane and of the requested size; vertical plane upright and in front of the arm
- [ ] Trajectory: TCP speed and acceleration within the limits, centripetal limit on curves, rest at every corner, retiming
- [ ] Joint plan: TCP (FK of the planned q) follows the path within `TRAJ_IK_TOL` with the pitch held, joints smooth (no jumps), retiming when joints too fast and capped, unreachable path rejected
- [ ] Approach/return: `joint_quintic` and `full_plan` continuous and back at the start
- [ ] Safety: no contacts along the demo shapes; HOME pose safe; rest <-> HOME accepted; a transition may not go lower than the start; shape motions use strict checks; a path out of limits or too low fails with the reason
- [ ] Controller: `reached()`, dry run moves nothing, one-shot run goes start -> HOME -> shape -> HOME -> start

## `simulation/trajectory_sim.py`
- [ ] For every shape (and one waypoint sequence) on both planes: plan, check, run in MuJoCo (position actuators at `LOOP_HZ`)
- [ ] Summary table: shape, length, duration, max TCP speed, IK error max, tracking error (RMS / max), path error (RMS / max), corners reached, checks
- [ ] CSV per shape and a summary CSV in `results/data/`
- [ ] Figures in `results/figures/`:
    - [ ] desired vs MuJoCo TCP in the drawing plane (one panel per shape, or a gallery)
    - [ ] tracking and path error over time
    - [ ] TCP speed profile and joint angles/speeds of one shape
- [ ] `--view` to watch it in the MuJoCo viewer, `--shape` to run only one

## `hardware/trajectory_demo.py`
- [ ] Follow the hardware rules of the brief:
    - [ ] begin from a known safe configuration (rest -> HOME)
    - [ ] small motions first (small `--size`, then the demo size)
    - [ ] conservative speeds (`TRAJ_V_MAX`)
    - [ ] targets inside the verified workspace (same region as `ik_hardware.py`)
    - [ ] joint limits checked before sending (`check_plan`)
    - [ ] ready to stop: confirmation before every move, Ctrl+C freezes
    - [ ] no configuration with self or table collision (MuJoCo check of the whole plan)
- [ ] `--dry-run` (plan and checks only), `--shape`, `--waypoints`, `--plane`, `--size`, `--z`, interactive menu at HOME
- [ ] Measured: tracking and path error from FK of the read q, corners reached, optional ruler measure of 2-3 corners
- [ ] CSV and figure `trajectory_<shape>_hw`
- [ ] Record the video for Deliverable 3 (physical robot visible, shape visible e.g. pen on paper or marked points, the motion, terminal with the controller running)

## `run.py`
- [ ] Phase 3 entries first:
    - [ ] trajectory tests
    - [ ] shapes in MuJoCo (all, or one with viewer)
    - [ ] final demonstration on the real arm (dry run without port)
- [ ] Keep Phase 2 and Phase 1 entries below

## Report (Phase 3 part of the final 4-6 page report)
1) Controller and trajectory generation:
- [ ] Cartesian command implementation (`target_position`, `target_pose`) and the 4-step loop
- [ ] Trajectory generation: Cartesian pieces, quintic time scaling, limits, centripetal limit, retiming, approach/return in joint space
- [ ] Controller implementation and parameter table (`TRAJ_*`, `LOOP_HZ`, tolerances)
- [ ] Safety chain before every move (checks, start/HOME rules)
2) Results and discussion:
- [ ] Simulation: summary table from `simulation/trajectory_sim.py` and one figure (desired vs MuJoCo shape + errors)
- [ ] Hardware: tracking/path error and reached corners on the real arm, one figure, ruler measure if done
- [ ] Position or trajectory accuracy: sim vs real
- [ ] Limitations: 5-DOF arm (pitch held, no free orientation), servo lag and backlash, calibration, speed limited by the bus
- [ ] Failure cases: shapes out of the workspace, too low, near the pan axis or the stretched-arm singularity (rejected by the checks)
- [ ] Possible improvements: closed-loop correction on the measured q, time-optimal retiming, obstacle-aware paths
