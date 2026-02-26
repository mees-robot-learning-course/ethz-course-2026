# HW2 SO-100 Tutorial — Installation Guide (Linux & Windows)

## Prerequisites

- Any virtual environment (conda / uv / venv) is recommended. **venv** is used here as an example.
- **Python 3.12** tested and recommended.
- **pip** (usually bundled with Python).
- **Git** (for downloading the repository).

**For Linux (Ubuntu/Debian):**
OpenGL / EGL drivers are needed by MuJoCo's viewer. They should already be installed, but you can explicitly install them with:
`sudo apt install libegl1-mesa-dev libgl1-mesa-dri libglvnd-dev`

**For Windows:**
Make sure `python` and `pip` are available in PowerShell (test with `python --version`). If not, add Python to your system PATH.

---

## 1. Create & Activate a Virtual Environment
From the root directory of the repository:
**Linux (Bash):**
`python3 -m venv mujoco`
`source mujoco/bin/activate`

> **Tip:** Every time you open a new terminal, re-activate with `source mujoco/bin/activate`.

**Windows (PowerShell):**
`python -m venv mujoco`
`.\mujoco\Scripts\Activate.ps1`

> **Tip:** If you get an execution policy error on Windows, run: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Installing Virtual Environment in Desktops at CAB H56 & H57:** 
For those who use the lab computers. The personal drive only has 10 GB sapce. The virtual environment requires 7-8 GB space. A viable solution is to install the venv in the /tmp folder (16 GB) and put your own code in the personal drive. You can write a bash script to install the venv and run it every time you log into a lab computer to restore the venv in \tmp.

---

## 2. Install the Package and Dependencies

Install the requirements and then the local `so101_gym` package in editable (development) mode — this automatically pulls in all required libraries:

`pip install -r hw2_robot_control_mdps/requirements.txt`
`pip install -e hw2_robot_control_mdps`

### What Gets Installed

| Package             | Purpose                                        |
|---------------------|-------------------------------------------------|
| `mujoco`            | Physics simulation engine + built-in viewer     |
| `gymnasium`         | RL environment API (used by `SO100TrackEnv`)    |
| `stable-baselines3` | RL algorithms (PPO) for training & evaluation   |
| `tensorboard`       | Training log visualization                      |

---

## 3. Verify the Installation

Quick smoke test — this should open the MuJoCo viewer with the SO-100 robot arm:

**Linux & Windows:**
`python scripts/interactive.py`

If a 3D viewer window opens showing the robot, everything is working. You can interact with the environment using:
- **Rotate:** Click and drag
- **Zoom:** Scroll wheel
- **Pan:** Shift + click and drag

Close the viewer window to exit.

---

## 4. Available Scripts

| Script                    | Description                                  | Command                          |
|---------------------------|----------------------------------------------|----------------------------------|
| `scripts/interactive.py`  | Launch the MuJoCo viewer to inspect the robot | `python scripts/interactive.py`  |
| `scripts/train.py`        | Train a PPO agent (16 parallel envs)          | `python scripts/train.py`        |
| `scripts/evaluate_rand_targets.py`     | Evaluate a trained policy in the viewer       | `python scripts/evaluate_rand_targets.py`     |

---

## 5. Monitor Training

Either in real-time or after training is finished, you can visualize metrics with TensorBoard:

**Linux & Windows:**
`tensorboard --logdir logs/ --port 6006`

Then open http://localhost:6006 in your browser.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'mujoco'` | Make sure your venv is activated and you ran `pip install -r requirements.txt`. |
| MuJoCo viewer doesn't open / EGL errors *(Linux)* | Install Mesa/EGL drivers: `sudo apt install libegl1-mesa-dev libgl1-mesa-dri libglvnd-dev` |
| `ERROR: could not create window` *(Linux)* | `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` |
| Viewer window is black *(Windows)* | Update your system GPU drivers. |
| `ModuleNotFoundError: No module named 'env'` | Run scripts from the `hw2_so100_tutorial/` directory, or make sure `PYTHONPATH=.` is set. Also check you ran `pip install -e .` |
| `python` not found *(Windows)* | Use `python3` instead, or add Python to your system PATH. |
| `pip install` fails *(Windows)* | Make sure the venv is activated (`.\mujoco\Scripts\Activate.ps1`). |
