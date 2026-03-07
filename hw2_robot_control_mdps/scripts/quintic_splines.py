import numpy as np
import time
import mujoco
import mujoco.viewer

from __init__ import XML_PATH
from utils import refresh_markers
from exercises.ex1 import build_keypoints, ik_track
from exercises.ex2 import generate_quintic_spline_waypoints


if __name__ == "__main__":
    keypoints = build_keypoints()
    keypoint_id = 0
    
    target_quat_wxyz = np.asarray([1, 0, 0, 0], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    data.mocap_pos[0] = keypoints[keypoint_id]

    site_name = "ee_site"
    num_waypoints = 5

    next_keypoint_id = (keypoint_id + 1) % len(keypoints)
    generate_waypoint_function = generate_quintic_spline_waypoints
    waypoints = generate_waypoint_function(keypoints[keypoint_id], keypoints[next_keypoint_id], num_waypoints)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        refresh_markers(viewer, keypoints)
        while viewer.is_running():
            next_keypoint_id = (keypoint_id + 1) % len(keypoints)
            data.mocap_pos[0] = keypoints[next_keypoint_id]
            waypoints = generate_waypoint_function(keypoints[keypoint_id], keypoints[next_keypoint_id], num_waypoints)
            refresh_markers(viewer, waypoints, radius=0.003, rgba=(0, 1, 1, 1), ngeom_start=len(keypoints))
            
            waypoint_id = 0
            while waypoint_id < num_waypoints:
                target_qpos = ik_track(model, data, site_name, waypoints[waypoint_id])
                data.qpos[:] = target_qpos
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.1)
                waypoint_id += 1
            keypoint_id = next_keypoint_id
            