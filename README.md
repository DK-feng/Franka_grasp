# A pybullet simulation environment for single object grasping tasks in which users can easily change the algorithm and target object used




## Using GG-CNN and Nut&Bolt to demonstrate
<p align='center'>
<img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/prasp_process.gif' width='600px'>
</p>
<br>

We placed a virtual camera between the two grippers of the robotic arm’s end effector, which can capture RGB-D images of the area directly in front of the end effector in real time. The small yellow goast cube which has no collision shape represents the target position. At the beginning of each task, we first randomly initialize the positions of the object and the target. Then, the robotic arm’s end effector is moved 40cm directly above the object, with some horizontal error (±5cm) to better validate the network. The end-effector then descends at a constant speed to 20 cm above the object. The camera will loss focus in real scenarios below this height. During this process, GG-CNN continuously updates the grasping parameters to obtain the optimal grasping solution. 

<br>

<p align='center'>
<img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/Visualize_grasp_parameters.gif' width='300px'>
</p>
<br>

Since GG-CNN is a closed-loop network which means the grasp parameters generated are changing continuously, we also add visulization code to demonstrate the update pracess. The performance is not very good dur to several reasons:    
1: The GG-CNN used is trained only on mini-jacquard, which has much less date than jacquard  
2: GG-CNN is a very simple network which cannot gain a very high performance  

<br>
<br>
<br>
      
## Problem may encounter:  
  #### Abnormal movement of the robotic arm (shown below)  
  <p align='center'>
  <img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/poor_inverse_kinematics.gif' width='400px'>  
  </p> 
  <p align='center'>
  <img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/error_movement.gif' width='400px'>  
  </p> 
  
  #### Reason for this:  
  PyBullet uses Damped Least Squares method to caculate inverse kinematics, however, this method tends to favor straight-line motion for the end effector.(shown below)  
  <p align='center'>
  <img src='https://github.com/DK-feng/Franka_grasp/blob/main/GIF_folder/straight_line_move.gif' width='400px'>
  </p> 
  
  #### Solution:
  If the distance between the current position and the target position is too great, it should be broken down into multiple smaller segments, with each segment being executed separately to reach the destination. This is also why we often see a discount factor applied to the target position in PyBullet, normally 0.05.














  
