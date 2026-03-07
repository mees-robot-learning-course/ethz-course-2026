import numpy as np
import time
import mujoco
import mujoco.viewer

from __init__ import TORQUE_CTRL_XML_PATH
from utils import refresh_markers
from exercises.ex1 import build_keypoints, ik_track
from exercises.ex2 import generate_quintic_spline_waypoints, pid_control


def update_tracking_error_history(tracking_error_history, target_qpos, max_length=10):
    current_qpos = data.qpos.copy()
    error = target_qpos - current_qpos
    
    if len(tracking_error_history) == 0:
        tracking_error_history = error.reshape(1, -1)
    else:
        tracking_error_history = np.vstack([tracking_error_history, error])
        if len(tracking_error_history) > max_length:
            tracking_error_history = tracking_error_history[1:]
    return tracking_error_history
    
def pid_callback(model, data):
    if len(tracking_error_history) == 0:
        data.ctrl[:] = 0
        return
    data.ctrl[:] = pid_control(tracking_error_history, model.opt.timestep)

    
if __name__ == "__main__":
    keypoints = build_keypoints()
    
    model = mujoco.MjModel.from_xml_path(str(TORQUE_CTRL_XML_PATH))
    data = mujoco.MjData(model)

    site_name = "ee_site"
    num_waypoints = 5

    total_waypoints = []
    keypoint_id = 0
    while keypoint_id < len(keypoints):
        next_keypoint_id = (keypoint_id + 1) % len(keypoints)
        waypoints = generate_quintic_spline_waypoints(keypoints[keypoint_id], keypoints[next_keypoint_id], num_waypoints)
        total_waypoints.append(waypoints)
        keypoint_id += 1
    total_waypoints = np.vstack(total_waypoints)

    # set to initial position
    target_qpos = ik_track(model, data, site_name, total_waypoints[0])
    data.qpos[:] = target_qpos
    mujoco.mj_forward(model, data)
    
    # pid
    tracking_error_history = np.array([])

    waypoint_id = 0
    mujoco.set_mjcb_control(pid_callback)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        refresh_markers(viewer, keypoints)
        refresh_markers(viewer, total_waypoints, radius=0.003, rgba=(0, 1, 1, 1), ngeom_start=len(keypoints))
        while viewer.is_running():
            target_qpos = ik_track(model, data, site_name, total_waypoints[waypoint_id])
            pid_callback.target_qpos = target_qpos
            data.mocap_pos[0] = total_waypoints[waypoint_id]
            tracking_error_history = update_tracking_error_history(tracking_error_history, target_qpos)
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
            if np.linalg.norm(total_waypoints[waypoint_id] - data.site(site_name).xpos) < 3e-2:
                tracking_error_history = np.array([])
                waypoint_id = (waypoint_id + 1) % len(total_waypoints)
            time.sleep(model.opt.timestep)
    mujoco.set_mjcb_control(None)