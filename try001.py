import numpy as np
import pybullet as p
import torch
import torch.nn.functional as F
import torch.nn as nn
import glob
import os
import time
import pybullet_data


class MyRobot():

    def __init__(self):
        self.ee_link = 11
        self.neutral_joint_state = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])  
        p.loadURDF("plane.urdf", [0, 0, -0.3])
        self.frankaId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
        p.resetBasePositionAndOrientation(self.frankaId, [0, 0, -0.3], [0, 0, 0, 1])
        self.basic_orientation = p.getQuaternionFromEuler([-np.pi, 0, 0])


        self.joint_active_ids = [0,1,2,3,4,5,6,9,10]
        self.lower_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, -5, -5]
        self.upper_limits = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973,  5,  5]
        self.joint_ranges = [ul-ll for ul,ll in zip(self.upper_limits,self.lower_limits)]
        self.rest_poses   = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.02, 0.02]


    def reset(self):
        self.set_action(self.neutral_joint_state)


    def set_action(self, actions:np.array):
        '''excute the actions'''
        if type(actions) == 'list':
            for joint_poses in actions:
                for i,j in enumerate(self.joint_active_ids):
                    p.setJointMotorControl2(self.frankaId,
                                    jointIndex=j,
                                    targetPosition=joint_poses[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)
        else:
            for i,j in enumerate(self.joint_active_ids):
                p.setJointMotorControl2(self.frankaId,
                                    jointIndex=j,
                                    targetPosition=actions[i],
                                    controlMode=p.POSITION_CONTROL,
                                    targetVelocity=0,
                                    force=500,
                                    positionGain=0.03,
                                    velocityGain=1)


    def setCameraAndGetPic(self, width:int=300, height:int=300):
        #设置虚拟摄像头,位于两爪子之间,图片为300*300,匹配神经网络输入
        ee_position = np.array(p.getLinkState(self.frankaId,self.ee_link)[0])
        hand_orien_matrix = p.getMatrixFromQuaternion(p.getLinkState(self.frankaId, self.ee_link)[1])
        z_vec = np.array([hand_orien_matrix[2],hand_orien_matrix[5],hand_orien_matrix[8]])

        camera_pos = ee_position + 0.02*z_vec  #加上一个z轴方向向量防止摄像头位于机械臂内部
        target_pos = ee_position + 0.3*z_vec     #摄像头视角总指向夹爪正前方

        view_matrix = p.computeViewMatrix(
            cameraEyePosition = camera_pos,
            cameraTargetPosition = target_pos,
            cameraUpVector = [0,1,0])
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=50.0,
            aspect=1.0,
            nearVal=0.01,
            farVal=20)
    
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        return width, height, rgbImg, depthImg, segImg


    def processing_ggcnnOutput(self,GGCNN_output):
        #处理GGCNN的输出,将张量变成有用的数据
        index = torch.argmax(GGCNN_output[0]).cpu().numpy()
        GGCNN_output = [x.to('cpu') for x in GGCNN_output]
        cos2 = GGCNN_output[1].cpu().numpy().reshape(-1)[index]
        sin2 = GGCNN_output[2].cpu().numpy().reshape(-1)[index]
        width = GGCNN_output[3].cpu().numpy().reshape(-1)[index]

        print(index)
        row_index, column_index = index//300 +1, index%300 +1
        print(row_index,'\t',column_index)
        print((150-column_index)/300,"\t",(row_index-150)/300,"\t",width)
        print('----------------------------------------------------------------')

        ee_position = p.getLinkState(self.frankaId,self.ee_link)[0]

        #根据抓取点在图像中不同的位置来不断微调desired_position,每次微调下降0.1cm
        desired_position = ee_position + 0.01*np.array([(150-column_index)/300, (row_index-150)/300, -0.1])
        theta = np.arctan(sin2/cos2)/2        #是弧度并非角度
        desired_orientation = p.getQuaternionFromEuler(np.array([np.pi, 0, theta]))     
        new_joints_state = p.calculateInverseKinematics(self.frankaId,self.ee_link,
                                                      desired_position,desired_orientation,
                                                      lowerLimits=self.lower_limits,
                                                      upperLimits=self.upper_limits,
                                                      jointRanges=self.joint_ranges,
                                                      restPoses=self.rest_poses)
        new_action = np.concatenate([new_joints_state[:7], np.array([width+0.02, width+0.02])])
        return new_action
  

    def get_action_from_position_and_orientation(self,position,orientation):
        #不改变爪子的宽度
        if len(orientation)==3:
            orientation = p.getQuaternionFromEuler(orientation) 
        finger_width = p.getJointState(self.frankaId, 9)[0] + p.getJointState(self.frankaId, 10)[0]
        new_joint_poses = p.calculateInverseKinematics(
            self.frankaId, self.ee_link, position, orientation
        )
        action = np.concatenate([np.array(new_joint_poses), np.array([finger_width/2, finger_width/2])])
        return action


    def move_to_position(self,position):
        orientation = p.getLinkState(self.frankaId, self.ee_link)[1]
        action = self.get_action_from_position_and_orientation(position, orientation)
        return action


    def set_finger_width(self, width):
        #设置爪子的宽度
        joint_poses = [p.getJointState[self.frankaId, i][0] for i in range(7)]
        action = np.concatenate([np.array(joint_poses), np.array([width/2, width/2])])
        return action
    

    def get_ee_height(self):
        height = p.getLinkState(self.frankaId, self.ee_link)[0][2]
        return height



class MyTask():

    def __init__(self, obj_path:str):
        self.obj_files = glob.glob(os.path.join(obj_path,'*.obj'))
        self.target_position = np.array([0, 0.5, -0.2])
        self.object_area_center = np.array([0.45,0.7,0.3])

    def reset(self):
        object_file = self.obj_files[np.random.randint(len(self.obj_files))]     #随即选择待抓取物体
        if 'Glock' in object_file:
            meshScale = [0.02, 0.02, 0.02]
        elif 'Mickey Mouse' in object_file:
            meshScale = [0.0004, 0.0004, 0.0004]
        if 'Nut & Bolt' in object_file:
            meshScale = [5,5,5]

        #随机物体的位置
        object_position = np.concatenate((np.random.uniform(low=0.4, high=0.5, size=2), np.array([-0.15])))
        object_orientation = np.random.uniform(low=-1,high=1,size=4)

        #创建物体并初始化位置
        visual_shape_id = p.createVisualShape(
                                    shapeType=p.GEOM_MESH,
                                    fileName=object_file,
                                    visualFramePosition=[0,0,0],
                                    meshScale=meshScale  
                                    )

        collision_shape_id = p.createCollisionShape(
                                    shapeType=p.GEOM_MESH,
                                    fileName=object_file,
                                    collisionFramePosition=[0,0,0],
                                    meshScale=meshScale   
                                    )
        
        objectId = p.createMultiBody(
                                    baseMass=1,
                                    baseCollisionShapeIndex=collision_shape_id,
                                    baseVisualShapeIndex=visual_shape_id,
                                    basePosition=[0.5, 0, 0.05],
                                    useMaximalCoordinates=True
                                    )

        p.resetBasePositionAndOrientation(objectId, object_position, object_orientation)



class GGCNN(nn.Module):
    #所用到的网络,输入深度图像,返回四个张量,内含抓取所需参数
    def __init__(self, input_channels=1):
        super().__init__()
        filter_sizes = [32, 16, 8, 8, 16, 32]       #其实就是channels
        kernel_sizes = [9, 5, 3, 3, 5, 9]
        strides = [3, 2, 2, 2, 2, 3]
        self.conv1 = nn.Conv2d(input_channels, filter_sizes[0], kernel_sizes[0], stride=strides[0], padding=3)
        self.conv2 = nn.Conv2d(filter_sizes[0], filter_sizes[1], kernel_sizes[1], stride=strides[1], padding=2)
        self.conv3 = nn.Conv2d(filter_sizes[1], filter_sizes[2], kernel_sizes[2], stride=strides[2], padding=1)
        self.convt1 = nn.ConvTranspose2d(filter_sizes[2], filter_sizes[3], kernel_sizes[3], stride=strides[3], padding=1, output_padding=1)
        self.convt2 = nn.ConvTranspose2d(filter_sizes[3], filter_sizes[4], kernel_sizes[4], stride=strides[4], padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(filter_sizes[4], filter_sizes[5], kernel_sizes[5], stride=strides[5], padding=3, output_padding=1)

        self.pos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.cos_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.sin_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)
        self.width_output = nn.Conv2d(filter_sizes[5], 1, kernel_size=2)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        with torch.no_grad():
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.relu(self.convt1(x))
            x = F.relu(self.convt2(x))
            x = F.relu(self.convt3(x))

            pos_output = self.pos_output(x) 
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

            return [pos_output, cos_output, sin_output, width_output]





if __name__ == "__main__":

    DEVICE = torch.device('cuda')

    client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0 ,0 ,-9.8)
    p.setRealTimeSimulation(1)

    robot = MyRobot()
    task = MyTask(obj_path='/home/dukaifeng/my_project/panda_project/obj_models')
    net = GGCNN()
    net.load_state_dict(torch.load('epoch_48_iou_0.88_statedict.pt'))

    net.to(DEVICE)

    robot.reset()
    task.reset()
    action = robot.get_action_from_position_and_orientation(task.object_area_center, robot.basic_orientation)
    robot.set_action(action)
    time.sleep(0.1)

    while True:
        height = robot.get_ee_height()
        while height > 0.1:
            _ ,_ , rgbImg, depthImg, _ = robot.setCameraAndGetPic()


            ggcnn_input = torch.tensor(depthImg).unsqueeze(0).unsqueeze(0).to(DEVICE)
            ggcnn_ouput = net(ggcnn_input)
            action = robot.processing_ggcnnOutput(ggcnn_ouput)
            robot.set_action(action)
            height = robot.get_ee_height()


        time.sleep(0.02)


        p.stepSimulation()

    p.disconnect()
