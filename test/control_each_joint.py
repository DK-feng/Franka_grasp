import pybullet as p
import pybullet_data
import time

client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


p.loadURDF("plane.urdf", [0, 0, -0.3])
frankaId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(frankaId, [0, 0, -0.3], [0, 0, 0, 1])


ee_Index = 11
joint_active_ids = [0,1,2,3,4,5,6,9,10]
p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)


# lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -5, -5]
# upper_limits = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  5,  5]
# joint_ranges = [ul-ll for ul,ll in zip(upper_limits,lower_limits)]
# rest_poses   = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.1, 0.1]


def get_poses():
    joint_1 = p.addUserDebugParameter('joint_0',  -2.8973,  2.8973,     0.00)
    joint_2 = p.addUserDebugParameter('joint_1',  -1.9628,  1.7628,     0.41)
    joint_3 = p.addUserDebugParameter('joint_2',  -2.8973,  2.8973,     0.00)
    joint_4 = p.addUserDebugParameter('joint_3',  -3.0718,  -0.0698,    -1.85)
    joint_5 = p.addUserDebugParameter('joint_4',  -2.8973,  2.8973,     0.00)
    joint_6 = p.addUserDebugParameter('joint_5',  -0.0175,  3.7525,     2.26)
    joint_7 = p.addUserDebugParameter('joint_6',  -2.8973,  2.8973,     0.79)
    joint_9 = p.addUserDebugParameter('joint_9',        0,      0.05,          0.1)
    joint_10 = p.addUserDebugParameter('joint_10',      0,      0.05,          0.1)

    return [joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7,joint_9,joint_10]


joint_poses = get_poses()


while True:

    for joint_indice,joint_pose in zip(joint_active_ids,joint_poses):
        p.setJointMotorControl2(frankaId,
                                jointIndex=joint_indice,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=p.readUserDebugParameter(joint_pose),
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
        
    p.stepSimulation()
    time.sleep(0.01)

p.disconnect()











