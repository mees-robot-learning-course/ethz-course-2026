import mujoco
import mujoco.viewer

from __init__ import *
from env.so100_tracking_env import SO100TrackEnv


if __name__ == "__main__":
    env = SO100TrackEnv(xml_path=XML_PATH, render_mode=None)
    mujoco.viewer.launch(env.model, env.data)