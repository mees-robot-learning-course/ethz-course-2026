from pathlib import Path
import numpy as np
import mujoco
from stable_baselines3.common.callbacks import BaseCallback


def quat_mul(q1, q2):
    q_out = np.zeros(4, dtype=np.float64)
    mujoco.mju_mulQuat(q_out, q1, q2)
    return q_out

def quat_conjugate(q):
    q_conj = np.zeros(4, dtype=np.float64)
    mujoco.mju_negQuat(q_conj, q)
    return q_conj

def quat_normalize(q):
    q_normalized = q.copy()
    mujoco.mju_normalize4(q_normalized)
    return q_normalized

def rot_mat_to_quat(mat):
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, mat.reshape(-1))
    return quat

def refresh_markers(viewer, points, radius=0.005, rgba=(1.0, 1.0, 0.0, 1.0), ngeom_start=0):
    user_scn = viewer.user_scn
    user_scn.ngeom = ngeom_start
    size = np.full(3, radius, dtype=np.float64)
    rgba_arr = np.asarray(rgba, dtype=np.float32)
    ident_mat = np.eye(3, dtype=np.float64).reshape(-1)
    for pos in points:
        if user_scn.ngeom >= user_scn.maxgeom:
            break
        pos_arr = np.asarray(pos, dtype=np.float64)
        mujoco.mjv_initGeom(
            user_scn.geoms[user_scn.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            pos_arr,
            ident_mat,
            rgba_arr,
        )
        user_scn.ngeom += 1


class EpisodeLoggingCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "episode" in info:
                # Log the tracking error at the end of the episode
                self.logger.record("rollout/ep_ee_tracking_error", info["ee_tracking_error"])
                    
        return True

class UpdateCheckpointCallback(BaseCallback):
    """Save the model every N PPO update steps (after each rollout) inside the run's log dir."""

    def __init__(self, save_path=None, name_prefix="model", save_freq_updates=10, verbose=0):
        super().__init__(verbose)
        self.save_path = Path(save_path) if save_path is not None else None
        self.name_prefix = name_prefix
        self.save_freq_updates = save_freq_updates
        self.update_counter = 0

    def _on_training_start(self) -> None:
        # Resolve to the actual run log dir (TensorBoard creates subfolders per run)
        if self.save_path is None:
            log_dir = self.logger.get_dir()
            if log_dir is None:
                raise ValueError("Logger directory is not set; cannot determine checkpoint path")
            self.save_path = Path(log_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        self.update_counter += 1
        if self.update_counter % self.save_freq_updates == 0:
            filename = f"{self.name_prefix}_{self.update_counter}"
            full_path = self.save_path / filename
            self.model.save(str(full_path))
            if self.verbose > 0:
                print(f"Saved checkpoint at {full_path}")
        return True
    
class KLAdaptiveLRCallback(BaseCallback):
    def __init__(self, target_kl=0.05, init_lr=1e-3, min_lr=1e-5, max_lr=1e-3,
                 up_factor=1.1, down_factor=0.7, tol=0.2):
        super().__init__()
        self.target_kl = target_kl
        self.lr = init_lr
        self.min_lr, self.max_lr = min_lr, max_lr
        self.up_factor, self.down_factor = up_factor, down_factor
        self.tol = tol  # acceptable relative band

    def _on_training_start(self):
        # Override PPO lr schedule with a mutable one we control
        self.model.lr_schedule = lambda _, lr=self.lr: lr
        self.model._update_learning_rate(self.model.policy.optimizer)
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = self.lr
        return True

    def _on_step(self) -> bool:
        # No per-step logic; required to satisfy BaseCallback abstract method
        return True

    def _on_rollout_end(self) -> bool:
        kl = self.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return True  # no KL recorded yet
        if kl > self.target_kl * (1 + self.tol):
            self.lr = max(self.min_lr, self.lr * self.down_factor)
        elif kl < self.target_kl * (1 - self.tol):
            self.lr = min(self.max_lr, self.lr * self.up_factor)
        self.model.lr_schedule = lambda _, lr=self.lr: lr
        self.model._update_learning_rate(self.model.policy.optimizer)
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = self.lr
        return True
    