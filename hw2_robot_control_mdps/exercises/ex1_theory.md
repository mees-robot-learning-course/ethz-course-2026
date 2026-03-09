# Exercise 1: Theoretical Questions

1. When the Lemniscate become too wide, the targets can fall outside of the robot's reachable workspace, which makes it impossible for the inverse kinematics solver to converge and may additionally result in singularities affecting stability as the robot arm extends to the edges of it's workspace.
2. The dt parameter scales the size of the update to the joint angles, so if dt is too large it can cause the solver to overshoot the target while too small a dt can make convergence to the target slow.
3. Numerical IK has the advantage of being generally applicable to a wide-range of robot geometries, while the closed-form solutios for analytical IK are only easily derived or derivable at all for specific robot geometries. The disadvantage of numerical IK is that it is iterative and thus slower and doesn't have the same convergence guarantees as more exact analytical IK.
4. For one our simple implementation doesn't account for rotation error, while off-the-shelf solvers of course do. Otherwise, among other benefits they provide, advanced off-the-shelf solvers handle joint limits and provide adaptive damping to dynamically handle singularities.

