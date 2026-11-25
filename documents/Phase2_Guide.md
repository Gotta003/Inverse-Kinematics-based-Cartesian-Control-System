# Phase 2 - Inverse Kinematics

The report asks for a numerical IK solver that finds the joint configuration reaching a desired Cartesian end-effector position or pose, with the main outcome being a reliable solver that reaches specified Cartesian targets in simulation and on real robot. The submission must include:
- [ ] Manipulator Jacobian (`src/jacobian.py`) with a personal check in `tests/test_jacobian.py` (In Report -> Method: Jacobian)
- [ ] Pseudoinverse/DLS IK (`src/inverse_kinematics.py`) with damped pseudo inverse and supporting functions located in folder `src/utils/` (In Report -> Method: DLS)
- [ ] Convergence criteria and parameters (`src/inverse_kinematics.py`) implementing the stop rules and the block of IK parameters in `src/utils/constants.py` (In report -> Parameter Table)
- [ ] Joint-limit handling (`src/utils/limits.py`) with an active clamping in `src/inverse_kinematics.py` (In Report -> Method: Joint limits)
- [ ] Singular and poorly conditioned configurations (various files) implementing adaptive damping, step control and $\sigma$_min monitoring with singularity tests and sweep (Paragraph for singularities and figure)
- [ ] >5 reachable Cartesian targets (`tests/test_ik.py`) Validation table with check in MuJoCo
- [ ] >1 unreachable or non-converging targets (`tests/test_ik.py`) Validation table with check in MuJoCo
- [ ] Validation in MuJoCo (`tests/test_ik.py` and `simulation/ik_validation.py`) Validation section
- [ ] Before hardware deployment (`hardware/ik_hardware.py) safe test on real arm with a short hardware paragraph in report

## Structural Elements
Five joints move the TCP with: 
- `shoulder_pan` rotating about the vertical;
- `shoulder_lift`, `elbow_flex`, `wrist_flex` rotate about parallel horizontal axes;
- `wrist_roll` rotates about the forearm
Then, the heading is not free, once the TCP position is fixed, only pitch and roll remain free, leading to these constraints:
1) Orientation cannot be treated like position, since a general 6-D pose is not reachable
2) The best approach is just position only with some constraints:
    - 3 constraints on the angles, since 2 DOF are redundant
    - 1 constraints for the tool pitch 
    - 1 constraint for  an orientation consistent with the arm

## Files
| Path | Content | Status | Role |
|---|---|---|---|
| `src/forward_kinematics.py` | FK with the PoE | from Phase 1 |  DONE |
| `src/jacobian.py` | space and TCP Jacobian, conditioning metrics | new | DONE |
| `src/inverse_kinematics.py` | IK solver, task modes, seeding, result type | new | DONE |
| `src/utils/helper.py` | add SO(3)/SE(3) helpers and the damped pseudoinverse | extended | Both |
| `src/utils/constants.py` | add a Jacobian block and an IK block | extended | Both |
| `src/utils/limits.py` | joint limits from the MuJoCo XML and the calibration file | new | DONE |
| `src/utils/sim_checks.py` | MuJoCo contact checks at a configuration and along a joint path | new | DONE |
| `tests/test_jacobian.py` | Jacobian checks | new | DONE |
| `tests/test_ik.py` | named Cartesian targets and IK property tests | new | DONE |
| `simulation/ik_validation.py` | large-scale validation, CSVs, figures, viewer | new | Miss Figures |
| `hardware/ik_hardware.py` | safe IK test on the real arm, with dry run | new | Flavio |
| `run.py`, `run.sh` | menu launcher and environment wrapper | extended |  |
| `results/data/`, `results/figures/` | outputs of the scripts | generated | Both |

## `src/jacobian.py` and `tests/test_jacobian.py`
- [x] Space Jacobian from screw axes implementing a `space_jacobian(q)` that accumulates the PoE and applies the $6\times6$ adjoint, reusing constants in `src/utils/constants.py`
- [x] TCP Jacobian implementation so a function `tcp_jacobian(q)` that outputs J_$\omega$ and J_p. It is done since MuJoCo has a function called `mj_jacSite` returns TCP Jacobian, so to test is required
- [x] Conditioning metrics computation in a function `jacobian_metrics(J), returning singular values, smallest singular value $\sigma$_min, manipulability and condition number
- [x] Do a `__main__` to test q=0 (output equal to screw axes) amd TCP Jacobian at q=0 with its singular values
- [x] In `tests/test_jacobian.py` we need to run 3 independent tests:
    - [x] Structure of J_s(0) equals screw axes
    - [x] Check output differences with FK (like vee check)
    - [x] Check J_TCP against mj_jacsite on gripper frame after mj_forward testing

## `src/utils/`
- [x] `helper.py` for math helpers and damped pseudoinverse
    - [x] Skew
    - [x] Vee
    - [x] Adjoint
    - [x] Inverse Homogeneous Transform
    - [x] SO(3) logarithm
    - [x] damped pseudoinverse, it should return the pseudoinverse, monitored singular value $\sigma$, the $\lambda$ used and null-space projector (this is to have an adaptive damping to stay away from singularities with step exact pseudoinverse, to preserve accuracy loss, SVD gives $\sigma$_min and allows truncating singular values that are zero by construction, can have a primary and a secondary task assigned)
- [x] `constants.py` add a Jacobian and IK block
- [x] `limits.py` for joint limits
    - [x] Read joint ranges directly from MuJoCo XML
    - [x] Convert calibrated range of the robot from its calibration file
    - [x] Intersection of the two ranges and shrink the result by safety margin
- [x] `sim_checks.py` MuJoCo collision checks (mainly for checking for validation and hardware)
    - [x] A function that returns the contacts at a configuration like floor or link to link
    - [x] A function that samples a oint-space straight segment and returns new contacts and the lowest TCP height along it

## `src/inverse_kinematics.py`
- [x] Result object (dataclass in Python) with: configuration, success flag, status (`converged`, `stalled`, `stationary`, `max_iters`), final position and orientation errors, iterations, restarts, $\sigma$_min at the solution and solve time and history
- [x] Implement functions to perform inverse kinematics having a target position (`ik_position`), a pitch (`ik_pitch`), a target pose (`ik_pose`) and a general `inverse_kinematics`
- [x] Tasks implementation:
    - [x] position
    - [x] pitch
    - [x] pose
- [x] Creation of priority levels:
    - [x] priority (Combining position and orientation)
    - [x] stacked (one stacked weighted DLS on J_p and w·J_o)
- [x] Check damping and fix parameters and constants to set a good threshold
- [x] Control step at each iteration
    - [x] Clip norms of the position and orientation errors, fed to solver and scale $\Delta q$, so no joint moves more than a max step
    - [x] If step increases a merit function performs a backtrack halving it
    - [x] Clip q to joint limits
- [x] Convergence criteria:
    - [x] `converged`: Position error is below a fraction of a mm and when the mode has an orientation task, the orientation too has to have that error;
    - [x] `stalled`: merit function has not improved for a number of iterations;
    - [x] `stationary`: step vanishes
    - [x] `max_iters`: max number of iterations reached
- [x] Joint Limits should use an active-set clamping, so if the step would push some joints past a limit, they should be moved exactly to the limit, remove Jacobian columns, re-solve and compensate the clamped motion and repeat until no new joint saturates
- [x] Seeding and restarts: if a configuration doesn't work try another seed for a distant target, but for consecutive a warm start converges in a few iterations and stays on same solution branch disabling restarts
- [x] Add `__main__` to perform a check

## `tests/test_ik.py`
- [x] Define targets (reachable, unreachable and non-converging)
- [x] Solve and check each target independently with a fixed seed to enable reproducibility
- [x] Report the table in `results/data/` with the result configuration format in `inverse_kinematics.py` to generate a table to insert in the report
- [x] Assertions for named targets for test
- [x] Checks on:
    - [x] Damped pseudoinverse
    - [x] Limits
    - [x] Warm-start round trip
    - [x] Cold-start round trip
    - [x] Priority vs Stacked
    - [x] Branch continuity
    - [x] Beyond joint limits
    - [x] Singularities

## `simulation/ik_validation.py`
- [x] Target sets in a box configuration everywhere inside the limits and a workspace only the ones collision free using `sim_checks.py`
- [x] For every target, run each mode, each method and two starts (warm and cold)
- [x] heck every solution in MuJoCo and print summary with statistics like success rate, error median, 95th percentile, maximum, iterations, time and failures. The class of failures may be [`pan_axis`, `near_singular`, `joint limit`, `near-miss`, `local minimum`]
- [x] Singularity experiment
- [x] Generate figures:
    - [x] Error CDFs (log scale with warm vs cold)
    - [x] Iterations and time per case
    - [x] Failure locations in radius-height plane
    - [x] Singular Experiment
- [x] Viewer mode to visualize the simulated arm

## `hardware/ik_hardware.py`
- [x] Recycle most of `hardware/fk_hardware.py` with a change in plan with a specified plan of poses and a random plan by rejection.
- [x] Check the full chain in MuJoCo and after that triggers the movement, if any check fails, print reasons, but nothing moves

## `run.py`
- [x] Extend the menu including the new modalities:
    - [x] Generalize the tests on all test files
    - [x] IK validation vs MuJoCo
    - [x] IK test on real arm

## Report
1) Method:
- [x] Jacobian with space Jacobian, TCP and verification
- [x] IK Solver with task modes, task priority and adaptive DLS
- [x] Convergence criteria and parameter table
- [x] Joint-limit handling: limits source, clamping and null-space centering
- [x] Singularities with adaptive damping, $\sigma$_min monitoring and step control
2) Validation:
- [x] Named target table from `tests/test_ik.py`
- [x] Compact summary of large-scale validation from `simulation/ik_validation.py`
- [x] Singularity figure and error figure
3) Failure cases and limitations + Hardware
- [x] Failure Classes
- [x] Limits of 5D arm
- [x] Planning and measured errors