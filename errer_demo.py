import pybullet as p
import time
import numpy as np
import pybullet_data



def setCameraAndGetPic(robot_id:int, width:int=300, height:int=300, client_id:int=0):
    
    finger1_pos = np.array(p.getLinkState(robot_id,linkIndex=9)[0])
    finger2_pos = np.array(p.getLinkState(robot_id,linkIndex=10)[0])
    hand_pos = np.array(p.getLinkState(robot_id,linkIndex=8)[0])
    hand_orien_matrix = p.getMatrixFromQuaternion(p.getLinkState(robot_id,linkIndex=8)[1])
    z_vec = np.array([hand_orien_matrix[2],hand_orien_matrix[5],hand_orien_matrix[8]])

    camera_pos = (finger1_pos + finger2_pos)/2 + z_vec
    target_pos = hand_pos + 5*z_vec

    view_matrix = p.computeViewMatrix(
        cameraEyePosition = camera_pos,
        cameraTargetPosition = target_pos,
        cameraUpVector = z_vec,
        physicsClientId=client_id)
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=50.0,
        aspect=1.0,
        nearVal=0.015,
        farVal=2,
        physicsClientId=client_id)
    
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    return width, height, rgbImg, depthImg, segImg





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

#正常
for i in range(10000):
    euler_angle = np.array([0,0,0])
    robot_end_orientation = p.getQuaternionFromEuler(euler_angle)
    joint_states = p.calculateInverseKinematics(robotId,11,[0.35,0.35,0.4],robot_end_orientation)

    p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=joint_states)
    p.stepSimulation()
    width, height, rgbImg, depthImg, segImg = setCameraAndGetPic(robot_id=robotId)
    time.sleep(1/200)


p.disconnect()



#更改欧拉角度机械臂位置不正确
'''for i in range(10000):
    euler_angle = np.array([0,np.pi,0])
    robot_end_orientation = p.getQuaternionFromEuler(euler_angle)
    joint_states = p.calculateInverseKinematics(robotId,11,[0.35,0.35,0.4],robot_end_orientation)

    p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=joint_states)
    p.stepSimulation()
    width, height, rgbImg, depthImg, segImg = setCameraAndGetPic(robot_id=robotId)
    time.sleep(1/200)


p.disconnect()'''




#放在循环外机械臂位置不正确
'''euler_angle = np.array([0,0,0])
robot_end_orientation = p.getQuaternionFromEuler(euler_angle)
joint_states = p.calculateInverseKinematics(robotId,11,[0.35,0.35,0.4],robot_end_orientation)

for i in range(10000):
    p.setJointMotorControlArray(robotId,joint_active_ids,p.POSITION_CONTROL,targetPositions=joint_states)
    p.stepSimulation()
    width, height, rgbImg, depthImg, segImg = setCameraAndGetPic(robot_id=robotId)
    time.sleep(1/200)


p.disconnect()'''

