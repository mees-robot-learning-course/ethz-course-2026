import numpy as np
import time
import mujoco
import mujoco.viewer

from __init__ import XML_PATH
from utils import refresh_markers
from exercises.ex1 import build_keypoints, ik_track


if __name__ == "__main__":
    keypoints = build_keypoints()
    keypoint_id = 0
    
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    data.mocap_pos[0] = keypoints[keypoint_id]

    site_name = "ee_site"

    with mujoco.viewer.launch_passive(model, data) as viewer:
        refresh_markers(viewer, keypoints)
        while viewer.is_running():
            target_qpos = ik_track(model, data, site_name, keypoints[keypoint_id])
            data.qpos[:] = target_qpos
            mujoco.mj_forward(model, data)
            viewer.sync()
            if np.linalg.norm(data.site(site_name).xpos - keypoints[keypoint_id]) < 5e-3:
                keypoint_id = (keypoint_id + 1) % len(keypoints)
                data.mocap_pos[0] = keypoints[keypoint_id]
            else:
                print("IK did not converge to the target within the threshold.")
                print(f"Tracking error: {np.linalg.norm(data.site(site_name).xpos - keypoints[keypoint_id]):.4f}")
            time.sleep(0.5)