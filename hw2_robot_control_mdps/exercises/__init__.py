import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
# Ensure the project root is importable so `env` resolves when running as a script
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

ASSETS_DIR = ROOT_DIR / "so101_gym" / "assets"
XML_PATH = ASSETS_DIR / "so100_pos_ctrl.xml"
TORQUE_CTRL_XML_PATH = ASSETS_DIR / "so100_torque_ctrl.xml"
LOG_DIR = ROOT_DIR / "logs"
EXP_NAME = "so100_tracking"
EXP_DIR = LOG_DIR / EXP_NAME