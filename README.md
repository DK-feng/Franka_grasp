Objective: Establish a robotic arm simulation environment for grasping tasks

Results:
![img](https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/result.gif#w10)

Problem haven't solved:
  1:Abnormal movement of the robotic arm (shown below)
  ![img](https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/poor_inverse_kinematics.gif)
  ![img](https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/error_movement.gif)

  reason for this: PyBullet uses Damped Least Squares method to caculate inverse kinematics, however, this method tends to favor straight-line motion for the end effector.(shown below)
  ![img](https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/straight_line_move.gif)
     
    
