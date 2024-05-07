![image](https://github.com/DK-feng/Franka_grasp/assets/137026721/45a4d9b5-673b-4773-b610-ad9a1a2e7c00)Objective: Establish a robotic arm simulation environment for grasping tasks

Results:
![img]('https://github.com/DK-feng/Franka_grasp/GIF_folder/result.gif')

Problem haven't solved:
  1:Abnormal movement of the robotic arm (shown below)
  ![img]("https://github.com/DK-feng/Franka_grasp/GIF_folder/poor_inverse_kinematics.gif")
  ![img]("https://github.com/DK-feng/Franka_grasp/GIF_folder/error_movement.gif")

    reason for this: PyBullet uses Damped Least Squares method to caculate inverse kinematics, however, this method tends to favor straight-line motion for the end effector.(shown below)
    ![img]("https://github.com/DK-feng/Franka_grasp/GIF_folder/straight_line_move.gif")
     
    
