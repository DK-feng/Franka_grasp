#Objective: Establish a robotic arm simulation environment for grasping tasks

#Results:   
<p align='center'>
<img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/result.gif' width='500px'>
</p>

#Problem haven't solved:  
  ##1:Abnormal movement of the robotic arm (shown below)  
  <p align='center'>
  <img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/poor_inverse_kinematics.gif' width='500px'>  
  </p> 
  <p align='center'>
  <img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/error_movement.gif' width='500px'>  
  </p> 
  
  ##Reason for this:  
  PyBullet uses Damped Least Squares method to caculate inverse kinematics, however, this method tends to favor straight-line motion for the end effector.(shown below)  
  <p align='center'>
  <img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/straight_line_move.gif' width='500px'>
  </p> 
