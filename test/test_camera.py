import pybullet as p
import numpy as np
import pybullet_data
import panda_gym




def setCameraAndGetPic(robot_id, ee_link:int=11, width:int=300, height:int=300, client_id:int=0):
    
    ee_position = np.array(p.getLinkState(robot_id,linkIndex=ee_link)[0])
    hand_orien_matrix = p.getMatrixFromQuaternion(p.getLinkState(robot_id,linkIndex=ee_link)[1])
    z_vec = np.array([hand_orien_matrix[2],hand_orien_matrix[5],hand_orien_matrix[8]])

    camera_pos = ee_position + 0.02*z_vec
    target_pos = ee_position + 0.25*z_vec

    view_matrix = p.computeViewMatrix(
        cameraEyePosition = camera_pos,
        cameraTargetPosition = target_pos,
        cameraUpVector = [0,1,0])
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=50.0,
        aspect=1.0,
        nearVal=0.001,
        farVal=100)
    
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL)
    
    return width, height, rgbImg, depthImg, segImg












client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


p.loadURDF("plane.urdf", [0, 0, -0.3])
frankaId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(frankaId, [0, 0, -0.3], [0, 0, 0, 1])

# objectId = p.loadURDF("franka_panda/panda.urdf")
# p.resetBasePositionAndOrientation(objectId, [1, 1, 0], [0, 0, 0, 1])


path = 'F:\Franka_grasp-main\obj_models\Mickey Mouse.obj'

visual_shape_id = p.createVisualShape(
    shapeType=p.GEOM_MESH,
    fileName=path,
    visualFramePosition=[0,0,0],
    meshScale=[0.0005,0.0005,0.0005]  
)


collision_shape_id = p.createCollisionShape(
    shapeType=p.GEOM_MESH,
    fileName=path,
    collisionFramePosition=[0,0,0],
    meshScale=[0.0005,0.0005,0.0005]   
)


objectId = p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collision_shape_id,
    baseVisualShapeIndex=visual_shape_id,
    basePosition=[0.5, 0, 0.05],
    useMaximalCoordinates=True
)


p.resetBasePositionAndOrientation(objectId, [0.5, 0, -0.15], [0, 1, 0, 1])








prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0



ee_Index = 11
joint_active_ids = [0,1,2,3,4,5,6,9,10]
# lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -5, -5]
# upper_limits = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  5,  5]
# joint_ranges = [ul-ll for ul,ll in zip(upper_limits,lower_limits)]
rest_poses   = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.1, 0.1]



for i,j in enumerate(joint_active_ids):
    p.resetJointState(frankaId, j, rest_poses[i])



p.setGravity(0, 0, -9.8)
t = 0.
p.setRealTimeSimulation(1)
trailDuration = 2



while True:
    t += 0.015
    p.stepSimulation()
    print(t)

    for i in range(3):
        ee_position = np.array([0.5+0.2*np.sin(t), 0.2*np.cos(t), 0.15])

        ee_orientation = p.getQuaternionFromEuler([-np.pi,0,0])

        joint_poses = p.calculateInverseKinematics(frankaId,ee_Index,ee_position,ee_orientation,)
        
        for i,j in enumerate(joint_active_ids):
            p.setJointMotorControl2(frankaId,
                                    jointIndex=j,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_poses[i],
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)

        _,_,_,depthImg,_ = setCameraAndGetPic(robot_id=frankaId)


    # ls = p.getLinkState(frankaId, ee_Index)
    # if (hasPrevPose):
    #     p.addUserDebugLine(prevPose, ee_position, [0, 0, 0.3], 1, trailDuration)
    #     p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    # prevPose = ee_position
    # prevPose1 = ls[4]
    # hasPrevPose = 1



    a = p.getLinkState(frankaId, ee_Index)[0]
    hand_orien_matrix = p.getMatrixFromQuaternion(p.getLinkState(frankaId,linkIndex=ee_Index)[1])
    z_vec = np.array([hand_orien_matrix[2],hand_orien_matrix[5],hand_orien_matrix[8]])
    # pos = [0.2*x+y for x,y in zip(z_vec,a)]
    pos = a
    if (hasPrevPose):
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 2, trailDuration)
    prevPose = pos
    hasPrevPose = 1


    # print(f"\n\n\n\n\n{pos}\n{type(pos)}\n\n\n\n")
    print(pybullet_data.getDataPath())





p.disconnect()











    # if i==100:
    #     current_joint_states = list(p.getJointStates(robotId,joint_active_ids))
    #     actual_joint_states = [x[0] for x in current_joint_states]
    #     print('-------------------------------------------------\n')
    #     print("ideal_joint_states:{}".format(joint_states[:7]))
    #     print("actual_joint_states:{}".format(actual_joint_states[:7]))
    #     print("ideal_finger_states:{}".format(joint_states[7:]))
    #     print("actual_finger_states:{}".format(actual_joint_states[7:]))
    #     print('\n-------------------------------------------------')


