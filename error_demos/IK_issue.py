import pybullet as p
import pybullet_data
import time
import numpy as np


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.loadURDF("plane.urdf", [0, 0, -0.3])
frankaId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(frankaId, [0, 0, -0.3], [0, 0, 0, 1])

p.setGravity(0,0,-10)
active_joint_index = [0,1,2,3,4,5,6,9,10]
ll = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -5, -5]
ul = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  5,  5]
jr = [ul-ll for ul,ll in zip(ul,ll)]
rp = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.1, 0.1]
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
ee_link = 11



p.setRealTimeSimulation(0)

time.sleep(5)

t = 0
while True:
    if t <= 300:
        for joint_pose,joint_index in zip(rp,active_joint_index):
            p.setJointMotorControl2(frankaId,
                                    jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_pose,
                                    targetVelocity=0,
                                    force=500,
                                    )
        t += 1

    else:
        target_pos = np.array([0.5,0,0.1])
        target_ori = p.getQuaternionFromEuler(np.array([0, np.pi,0]))


        keys = p.getKeyboardEvents()
        if ord('1') in keys:
            target_pos = np.array([0.5, 0.5, 0.1])
        if ord('2') in keys:
            target_pos = np.array([-0.5,0.5,0.1])
        if ord('3') in keys:
            target_pos = np.array([-0.5, -0.5, 0.1])
        if ord('4') in keys:
            target_pos = np.array([0.5,-0.5,0.1])
        if ord('6') in keys:
            target_pos = np.array([0.6, 0.1, 0.1])
        if ord('7') in keys:
            target_pos = np.array([0.6,-0.1,0.1])
        if ord('8') in keys:
            target_pos = np.array([0.4, 0.1, 0.1])
        if ord('9') in keys:
            target_pos = np.array([0.4,-0.1,0.1])


        joint_poses = p.calculateInverseKinematics(frankaId,
                                                    ee_link,
                                                    targetPosition=target_pos,
                                                    targetOrientation=target_ori,
                                                    jointDamping=jd,
                                                    lowerLimits=ll,
                                                    upperLimits=ul,
                                                    jointRanges=jr,
                                                    restPoses=rp,
                                                    maxNumIterations=1000,
                                                    residualThreshold=0.1)[:7]


        for joint_pose,joint_index in zip(joint_poses,active_joint_index):
            p.setJointMotorControl2(frankaId,
                                    jointIndex=joint_index,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_pose,
                                    targetVelocity=0,
                                    force=500,
                                    )


        current_poses = [p.getJointState(frankaId,i)[0] for i in active_joint_index]
        print("current poses:{}".format(current_poses))
        print("ideal poses:{}".format(joint_poses))
        print('\n\n')



    p.stepSimulation()
    time.sleep(0.01)


p.disconnect()