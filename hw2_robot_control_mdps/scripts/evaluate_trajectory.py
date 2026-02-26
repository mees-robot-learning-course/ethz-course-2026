import argparse
import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO

from __init__ import *
from utils import refresh_markers
from env.so100_tracking_env import SO100TrackEnv
from exercises.ex1 import build_keypoints


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PID on SO100 tracking")
    parser.add_argument("--load_run", type=str, default="1",
                        help="training id")
    parser.add_argument("--checkpoint", type=str, default="500",
                        help="checkpoint id")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device (cpu or cuda)")
    return parser.parse_args()
     
def policy_callback(model, data):
    step_count = getattr(policy_callback, "step_count", 0)
    keypoint_id = getattr(policy_callback, "keypoint_id", 0)

    ee_tracking_error = np.linalg.norm(data.site("ee_site").xpos - data.mocap_pos[0])
    if ee_tracking_error < 0.05:
        keypoint_id += 1
        if keypoint_id >= len(keypoints):
            keypoint_id = 0
    data.mocap_pos[0] = keypoints[keypoint_id]

    if step_count > play_episode_length * env.ctrl_decimation:
        step_count = 0
    
    if step_count % env.ctrl_decimation == 0:
        obs = env._get_obs()
        action, _states = rl_model.predict(obs, deterministic=True)
        data.ctrl[:] = env._process_action(action)

    policy_callback.step_count = step_count + 1
    policy_callback.keypoint_id = keypoint_id


if __name__ == "__main__":
    args = parse_args()
    policy_path = EXP_DIR / f"so100_tracking_{args.load_run}" / f"model_{args.checkpoint}.zip" 
    
    env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
    play_episode_length_s = 5
    play_episode_length = int(play_episode_length_s / env.ctrl_timestep)

    keypoints = build_keypoints(count=20, width=0.2, x_offset=0.3, z_offset=0.25)
    env.data.mocap_pos[0] = keypoints[0]

    print(f"Loading model from {policy_path}...")
    rl_model = PPO.load(policy_path, device=args.device)

    mujoco.set_mjcb_control(policy_callback)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        refresh_markers(viewer, keypoints)
        while viewer.is_running():
            mujoco.mj_step(env.model, env.data)
            viewer.sync()
            time.sleep(env.model.opt.timestep)
    mujoco.set_mjcb_control(None)