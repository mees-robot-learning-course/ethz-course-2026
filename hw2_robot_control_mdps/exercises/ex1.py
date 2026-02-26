import numpy as np
import mujoco


def get_lemniscate_keypoint(t, a=0.2):
    """
    TODO:
    Generate a set of keypoints using Lemniscate of Bernoulli (infinity sign) in the Y-Z plane.
        The formula is: y = a * cos(t) / (1 + sin(t)^2)
                        z = a * cos(t) * sin(t) / (1 + sin(t)^2)
    For interest, you can learn about Lemniscate of Bernoulli on wikipedia: https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli
    
    Args:
        t (float or np.ndarray): Time scales from 0 to 2π to generate keypoints.
        a (float): Scaling factor for the size of the lemniscate.
        
    Returns:
        y (float or np.ndarray): y coordinates of the keypoint on the lemniscate.
        z (float or np.ndarray): z coordinates of the keypoint on the lemniscate.
    """
    raise NotImplementedError()

def build_keypoints(count=16, width=0.25, x_offset=0.3, z_offset=0.25):
    """TODO:
    Build a set of keypoints (x, y, z) along the lemniscate trajectory.
    Steps:
    1. Generate `count` linearly spaced time values `t` between 0 and 2π (exclusive).
    2. For each time value `t`, compute the corresponding (y, z) coordinates using `get_lemniscate_keypoint(t, a=width)`.
    3. Combine the (y, z) coordinates with a fixed x coordinate (x_offset) and additive z_offset to create 3D keypoints in the format [x_offset, y, z + z_offset].
    4. Return the keypoints as a NumPy array of shape (count, 3).

    Args:
        count (int): Number of keypoints to generate along the trajectory.
        width (float): Scaling factor for the size of the lemniscate.
        x_offset (float): Fixed x coordinate for all keypoints.
        z_offset (float): Offset to add to the z coordinate of all keypoints.

    Returns:
        np.ndarray: Array of shape (count, 3) containing the generated keypoints.
    """
    raise NotImplementedError()

def ik_track(model, data, site_name, target_pos,
             damping=1e-3, pos_gain=2.0, dt=0.1, max_iters=2000):
    """TODO:
    Implement an IK tracking function that computes the joint configuration to reach a target end-effector position. We ignore orientation tracking for simplicity.
    The function should iteratively update the joint configuration using the Jacobian of the end-effector until it reaches the target within a specified tolerance 
    or exceeds the maximum number of iterations. We use the Damped Least Squares method to handle singularities in the Jacobian. For interest, you can learn about 
    Damped Least Squares method on wikipedia: https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm

    Steps:
    1. Store the original joint configuration (qpos) to restore later.
    2. For a maximum number of iterations:
        a. Compute the current end-effector position and orientation using forward kinematics (mj_kinematics).
        b. Calculate the position error (target_pos - current_pos).
        c. If the position error is below a certain threshold (e.g., 1e-3), break the loop as we have reached the target.
        d. Compute the Jacobian of the end-effector using mj_jacSite.
        e. Use the Damped Least Squares to compute the change in joint configuration (qdot) that would reduce the error.
        f. Update the joint configuration (qpos) using the output from the Damped Least Squares method.
    3. Restore the original joint configuration and return the target joint configuration that was computed.

    Args:
        model: MuJoCo model object.
        data: MuJoCo data object.
        site_name: Name of the end-effector site to track.
        target_pos: Desired position of the end-effector (3D vector).
        damping: Damping factor for the Damped Least Squares method to handle singularities.
        pos_gain: Gain factor for the position error in the control signal.
        dt: Time step for updating the joint configuration.
        max_iters: Maximum number of iterations to attempt for reaching the target.

    Returns:
        np.ndarray: Target joint configuration (qpos) that achieves the desired end-effector position.
    """
    num_joints = model.nv
    # Store the original joint configuration to restore later
    original_qpos = data.qpos.copy()

    for i in range(max_iters):
        # use forward kinematics to update current end-effector position: data.site(site_name).xpos
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        # TODO: compute end-effector position error
        err_pos = ...

        # TODO: check if the 2-norm of the position error is within a small threshold (1e-3), if yes, break the loop
        ...
        
        # Get the Jacobian of the end-effector using mj_jacSite.
        jacp = np.zeros((3, num_joints)) # position Jacobian
        jacr = np.zeros((3, num_joints)) # orientation Jacobian
        mujoco.mj_jacSite(model, data, jacp, jacr, model.site(site_name).id)
        J = np.vstack([jacp, jacr])  # shape (6, nv)

        # TODO: compute the change in joint configuration (qdot) using Damped Least Squares method to reduce the position error
        # Damped least squares: qdot = J^T @ (J @ J^T + damping * I)^-1 @ weighted_err
        # Hint: damping * I is a 6x6 matrix with damping on the diagonal, and weighted error is a 6D vector (3 for pos, 3 for rot) of the form 
        # [pos_gain * err_pos, rot_gain * err_rot]. Since we are ignoring orientation tracking, you can set the rotational part of the weighted error to zero.
        # Instead of directly computing the matrix inverse (which can be numerically unstable), you should use np.linalg.solve to solve the 
        # linear system (J @ J^T + damping * I) x = weighted_err for x, and then compute qdot = J^T @ x. This is more stable and efficient than computing the inverse.
        qdot = ...

        # optional clamp to avoid overshoot
        qdot = np.clip(qdot, -2.0, 2.0)

        # Update the joint configuration (qpos) using the output from the Damped Least Squares method
        data.qvel[:] = 0.0
        data.qpos[:] += qdot * dt

    # If exiting the loop without reaching the target, print a warning message
    if i >= max_iters - 1 and np.linalg.norm(err_pos) >= 5e-3:
        print("Warning: IK did not converge within the iteration limit.")
        print(f"Final position error: {np.linalg.norm(err_pos):.4f}")

    # Restore the original joint configuration and return the target joint configuration
    target_qpos = data.qpos.copy()
    data.qpos[:] = original_qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_forward(model, data)
    return target_qpos
