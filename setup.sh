#!/usr/bin/env bash
#Run:
#   chmod +x setup.sh
#   ./setup.sh

set -e
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info() {
    echo -e "${CYAN}[INFO]${RESET} $*";
}

success() {
    echo -e "${GREEN}[OK]${RESET} $*";
}

warn() {
    echo -e "${YELLOW}[WARN]${RESET} $*";
}

error() {
    echo -e "${RED}[ERROR]${RESET} $*";
}

# 1) Check Conda & Activate 'lerobot' Environment
info "Checking Conda Installation..."
CONDA_EXEC=$(command -v conda || true)
if [ -z "$CONDA_EXEC" ]; then
    for path in "$HOME/miniforge3/bin/conda" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
        if [ -x "$path" ]; then
            CONDA_EXEC="$path"
            break
        fi
    done
fi

if [ -z "$CONDA_EXEC" ]; then
    error "Conda executable not found in PATH or standard dirs."
    exit 1
fi
success "Found conda at: $CONDA_EXEC"
CONDA_BASE=$("$CONDA_EXEC" info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

ENV_NAME="lerobot"
info "Locate conda environment '$ENV_NAME' exists..."
ENV_PATH=$(conda info --envs | awk '{print $NF}' | grep -E "/${ENV_NAME}$" | head -n 1 || true)

if [ -z "$ENV_PATH" ] || [ ! -d "$ENV_PATH" ]; then
    error "Could not find a conda environment path matching '$ENV_NAME'."
    exit 1
fi

info "Found environment at: $ENV_PATH"
info "Activating conda environment '$ENV_NAME'..."
conda activate "$ENV_PATH"
success "Environment '$ENV_NAME' activated"

#2) Python Version
info "Verifying Python inside '$ENV_NAME'..."
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ $PYTHON_MAJOR -lt 3 || ($PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 12) ]]; then
    error "Python 3.12+ required inside '$ENV_NAME' (found $PYTHON_VERSION)."
    exit 1
fi
success "Active Python version is $PYTHON_VERSION"

# 3) Install Requirements
if [ -f "requirements.txt" ]; then
    info "Installing packages from requirements.txt..."
    pip install -r requirements.txt
    success "Dependencies installed from requirements.txt."
else
    warn "requirements.txt not found in current directory."
fi

# 4) Run Package Verifier
if [ -f "check_packages.py" ]; then
    info "Running check_packages.py..."
    python check_packages.py
    success "Package verification passed."
else
    warn "check_packages.py not found."
fi

# 5) Directory Structure
info "Project Skeleton"
# Directories from Deliverable 1
mkdir -p src
mkdir -p simulation
mkdir -p hardware
mkdir -p tests
mkdir -p results/figures
mkdir -p results/data

# Placeholder files as specified in the rubric
touch src/forward_kinematics.py
touch src/jacobian.py
touch src/inverse_kinematics.py
touch src/trajectory.py
touch src/robot_control.py
touch simulation/mujoco_demo.py
touch hardware/lerobot_demo.py
touch tests/test_fk.py
touch tests/test_ik.py

if [ ! -f "README.md" ]; then
    cat << 'EOF' > README.md

# LeRobot SO-101 Inverse Kinematics & Cartesian Control

## Team Members
- Member 1
- Member 2
- Member 3

## Project Description
Inverse-kinematics-based Cartesian control system for the LeRobot SO-101 robotic arm.

## Installation / Setup
Run setup script:
```bash
chmod +x setup.sh
./setup.sh