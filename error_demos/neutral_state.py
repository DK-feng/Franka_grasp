import pybullet as p
import time
import numpy as np
import pybullet_data



physicsClient = p.connect(p.GUI)
# p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0,0,-10)
planeID = p.loadURDF("plane.urdf")
startPos = np.array([0,0,0.01],dtype=np.float32)
startOrientation = p.getQuaternionFromEuler([0,0,0])
robotId = p.loadURDF("franka_panda/panda.urdf",startPos,startOrientation,useFixedBase=True)
p.resetBasePositionAndOrientation(robotId,startPos,startOrientation)


joint_active_ids = np.array([0,1,2,3,4,5,6,9,10])





#不先设置中立位置则不正常
x = 0
for i in range(10000):
    euler_angle = np.array([0,np.pi,0])
    robot_end_orientation = p.getQuaternionFromEuler(euler_angle)
    joint_states = p.calculateInverseKinematics(robotId,11,[0.35,0.35,0.4],robot_end_orientation)

    p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=joint_states)
    p.stepSimulation()
    time.sleep(1/200)

    current_joint_state = [p.getJointState(robotId,x)[0] for x in joint_active_ids]
    if x == 200:
        print('\n\n\n')
        print(joint_states)
        print(current_joint_state)
        print('\n\n\n')
        
    print(x)
    x += 1

p.disconnect()










# neutral_joint_state = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])  
# p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=neutral_joint_state)
# p.stepSimulation()


# #设置中立位置则正常
# x = 0
# for i in range(10000):
#     euler_angle = np.array([0,np.pi,0])
#     robot_end_orientation = p.getQuaternionFromEuler(euler_angle)

#     if i <= 100:
#         joint_states = neutral_joint_state
#     else:
#         joint_states = p.calculateInverseKinematics(robotId,11,[0.5,0.5,0.4],robot_end_orientation)
 
#     p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=joint_states)
#     p.stepSimulation()
#     time.sleep(1/200)

#     current_joint_state = [p.getJointState(robotId,x)[0] for x in joint_active_ids]
#     if x == 200:
#         print('\n\n\n')
#         print(joint_states)
#         print(current_joint_state)
#         print('\n\n\n')
        
#     print(x)
#     x += 1
    
# p.disconnect()



































# 放在循环外机械臂位置不正确
# euler_angle = np.array([0,0,0])
# robot_end_orientation = p.getQuaternionFromEuler(euler_angle)
# joint_states = p.calculateInverseKinematics(robotId,11,[0.35,0.35,0.4],robot_end_orientation)

# for i in range(10000):
#     p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=joint_states)
#     p.stepSimulation()

#     time.sleep(1/200)


# p.disconnect()



