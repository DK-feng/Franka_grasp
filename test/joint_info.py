import pybullet as p
import pybullet_data




p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


p.loadURDF("plane.urdf", [0, 0, -0.3])
frankaId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(frankaId, [0, 0, -0.3], [0, 0, 0, 1])

joint_type_dic = ['JOINT_REVOLUTE', 'JOINT_PRISMATIC', 'JOINT_SPHERICAL', 'JOINT_PLANAR', 'JOINT_FIXED']



for i in range(p.getNumJoints(frankaId)):
    info = p.getJointInfo(frankaId, i)
    print("joint index:{}".format(info[0]))
    print("joint type:{}".format(joint_type_dic[info[2]]))
    print("joint name:{}".format(info[1]))
    print("joint damping:{}".format(info[6]))
    print("joint friction:{}".format(info[7]))
    print("joint low limit:{}".format(info[8]))
    print("joint high limit:{}".format(info[9]))
    print("joint max force:{}".format(info[10]))
    print("joint max velocity:{}".format(info[11]))
    print('\n\n')


p.disconnect()