import numpy as np
from gymnasium import spaces
from panda_gym.envs.core import PyBulletRobot
from panda_gym.envs.core import Task
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.pybullet import PyBullet
import pybullet as p
import torch
import torch.nn.functional as F
import torch.nn as nn
import time



class MyRobot(PyBulletRobot):

    def __init__(self,sim):
        super().__init__(
            sim,
            body_name='franka',
            file_name='franka_panda/panda.urdf',
            base_position=np.array([0,0,0]),
            joint_indices=np.array([0,1,2,3,4,5,6,9,10]),
            action_space=spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32),
            joint_forces=np.array([187.0, 187.0, 187.0, 187.0, 120.0, 120.0, 120.0, 170.0, 170.0])
        )
        self.finger_indices = np.array([9,10])
        self.ee_link = 11
        self.neutral_joint_state = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])  

        self.sim.set_lateral_friction(self.body_name, self.finger_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.finger_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.finger_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.finger_indices[1], spinning_friction=0.001)
        self.sim.create_plane(z_offset=-0.3)
        self.sim.create_table(length=2,width=2,height=0.3)

    def set_action(self,action):
        action = action.copy()
        if type(action)==list:
            for action in action:
                self.control_joints(action)
                time.sleep(0.02)
        else:
            self.control_joints(action)

    def get_obs(self):
        #返回7位array,前3位置,后4四元数
        ee_position = np.array(self.sim.get_link_position(self.body_name, self.ee_link))    
        ee_orientation = np.array(self.sim.get_link_orientation(self.body_name, self.ee_link))
        observation = np.concatenate((ee_position,ee_orientation))    #前三位置,后四四元数
        return observation
    
    def reset(self):
        self.set_joint_angles(self.neutral_joint_state)

    def _get_ee_position(self):
        ee_position = self.sim.get_link_position(self.body_name,self.ee_link)
        return ee_position
    
    def _get_ee_orientation(self):
        ee_orientation = self.sim.get_link_orientation(self.body_name, self.ee_link)
        return ee_orientation
    
    def _get_finger_width(self):
        finger1 = self.sim.get_joint_angle(self.body_name, self.finger_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.finger_indices[1])
        return finger1 + finger2
    
    def _inverse_kinematics(self,desired_ee_position,desired_ee_orientation):
        #更方便的逆运动学
        action =  self.sim.inverse_kinematics(self.body_name, self.ee_link,
                                              desired_ee_position, desired_ee_orientation)
        return action

    def _vertical_move(self, distance:int):
        #仅仅竖直移动,单位m
        current_ee_position = self._get_ee_position()
        desired_ee_position = current_ee_position + np.array([0,0,distance])
        desired_ee_orientation = self._get_ee_orientation()
        action = self._inverse_kinematics(desired_ee_position,desired_ee_orientation)
        finger_width = self._get_finger_width()/2
        action = np.concatenate([action[:7], np.array([finger_width,finger_width])])
        return action

    def _grip(self,distance:float):
        #收紧/松开机械臂的爪子,单位m
        current_joint_angles = np.array([self.get_joint_angle(i) for i in self.joint_indices])
        new_joint_angles = current_joint_angles + np.array([0,0,0,0,0,0,0,distance,distance])
        return new_joint_angles

    def setCameraAndGetPic(self, width:int=300, height:int=300, client_id:int=0):
        #设置虚拟摄像头,位于两爪子之间,图片为300*300,匹配神经网络输入
        ee_position = self.get_link_position(self.ee_link)
        hand_orien_matrix = p.getMatrixFromQuaternion(self.sim.get_link_orientation(self.body_name, self.ee_link))
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
        row_index, column_index = index//300 +1, index%300 +1

        ee_position = self.sim.get_link_position(self.body_name,self.ee_link)

        #根据抓取点在图像中不同的位置来不断微调desired_position,每次微调下降0.1cm


        desired_position = ee_position + 0.01*np.array([(150-column_index)/300, (row_index-150)/300, -0.1])

        theta = np.arctan(sin2/cos2)/2        #是弧度并非角度
        desired_orientation = p.getQuaternionFromEuler(np.array([np.pi, 0, theta]))     
        new_joints_state = self._inverse_kinematics(desired_position, desired_orientation)
        new_action = np.concatenate([new_joints_state[:7], np.array([width+0.02, width+0.02])])
        return new_action
  


class MyTask(Task):

    def __init__(self,sim):
        #在特定区域随机创建object和target
        super().__init__(sim)
        self.sim.create_box(body_name='object',
                            half_extents=np.array([0.03,0.03,0.03]),
                            mass=1,
                            position=np.array([0.35,0.35,0.03]),
                            rgba_color=np.array([255,97,0,80]))
        self.sim.create_box(body_name='target',
                            half_extents=np.array([0.03,0.03,0.03]),
                            mass=0,
                            position=np.array([-0.5,-0.5,0.03]),
                            ghost=True,
                            rgba_color=np.array([255,255,255,100]),)

    def reset(self):
        #随机生成物体和目标点 '''---------------------------------------------------------------------------
        # object_location = np.random.uniform(low=0.25, high=0.45, size=2)
        object_position = np.concatenate(((np.array([0.6,0])), np.array([0.03])))
        object_orientation = np.random.uniform(low=-1,high=1,size=4)
        target_location = np.random.uniform(low=-0.6, high=-0.2, size=2)
        target_position = np.concatenate((target_location, np.array([0.03])))
        target_orientation = np.random.uniform(low=-1,high=1,size=4)
        self.sim.set_base_pose(body='object', position=object_position, orientation=object_orientation)
        self.sim.set_base_pose(body='target',position=target_position,orientation=target_orientation)

    def get_obs(self):
        #返回物体的位置和目标点状态,前3位置后4姿态
        object_position = self.sim.get_base_position(body='object')
        object_orientation = self.sim.get_base_orientation(body='object')
        target_position = self.sim.get_base_position(body='target')
        target_orientation = self.sim.get_base_orientation(body='target')
        observation = np.concatenate((object_position,object_orientation,target_position,target_orientation))
        return observation  
    
    def get_achieved_goal(self):
        #返回物体目前的位置,前3位置后4姿态
        position = self.sim.get_base_position(body='object')
        orientation = self.sim.get_base_orientation(body='object')
        observation = np.concatenate((position,orientation))
        return observation 
    
    def get_goal(self):
        #返回target的位置和姿态,3+4,np.array
        position = self.sim.get_base_position(body='target')
        orientation = self.sim.get_base_orientation(body='target')
        desired_info = np.concatenate((position,orientation))
        return desired_info

    def is_success(self, achieved_goal, desired_goal):
        #判断是否成功,输入7+7
        loss = np.sum((achieved_goal-desired_goal)**2/2)
        return np.array(loss<=1.0, dtype=bool)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        loss = np.sum((achieved_goal-desired_goal)**2/2)
        return np.array(loss, dtype=np.float32)



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



class MyRobotTaskEnv(RobotTaskEnv):

    def __init__(self,render_mode='human'):
        self.DEVICE = torch.device('cuda')
        sim = PyBullet(render_mode=render_mode)
        self.robot = MyRobot(sim)
        self.task = MyTask(sim)
        self.net = GGCNN().to(self.DEVICE)
        self.net.load_state_dict(torch.load('epoch_48_iou_0.88_statedict.pt'))  #加载训练好的权重
        self.stage = 0
        self.time = 0
        super().__init__(self.robot,self.task)

    def get_action(self):
        #调整阶段
        self.time += 0.1
        if self.stage == 0:
            _,_,_,depthImg,_ = self.robot.setCameraAndGetPic()
            GGCNN_input = torch.tensor(depthImg).unsqueeze(0).unsqueeze(0).to(self.DEVICE)
            GGCNN_output = self.net(GGCNN_input)
            action = self.robot.processing_ggcnnOutput(GGCNN_output)
            ee_position = self.robot._get_ee_position()
            if ee_position[2] <= 0.15:       #只在摄像头高度大于15cm时微调,若小于这个距离,摄像头会失焦
                self.stage = 1
            print('0000000000000000000000000000000')     
            return action

        if self.stage == 1:
            action_1 = self.robot._vertical_move(-0.001)     #下降14cm
            if self.robot._get_ee_position()[2] < 0.02:
                self.stage = 2
                self.stage_2_num = 0
            print('1111111111111111111111111111111111111111111')
            return action_1

        if self.stage == 2:
            self.stage_2_num += 1
            action_2 = self.robot._grip(-0.0001)
            if self.stage_2_num >= 201:
                self.stage = 3
                self.stage_3_num = 0
            print('2222222222222222222222222222222222222222222')
            return action_2
        
        if self.stage == 3:
            action_3 = self._delivery() 
            self.stage = 4
            print('333333333333333333333333333333333333333333333')
            return action_3
        
        if self.stage == 4:
            action_4 = self.robot._vertical_move(-0.001)
            if self.robot._get_ee_position()[2] < 0.02:
                self.stage = 5
            print('4444444444444444444444444444444444444444444')
            return action_4
        
        if self.stage == 5:
            action_5 = self.robot._grip(0.002)
            self.stage = 6
            print('55555555555555555555555555555555555555555555')
            return action_5

        if self.stage == 6:
            action_6 = self.robot._vertical_move(0.01)    
            if self.robot._get_ee_position()[2] > 0.3:
                self.stage = 0
            print('66666666666666666666666666666666666666666666')
            return action_6


    def _delivery(self):
        #送到目标点上空20cm并调整姿态
        goal_info = self.task.get_goal()
        desired_position = goal_info[:3] + np.array([0,0,0.2])
        desired_orientation = goal_info[3:]
        action = self.robot._inverse_kinematics(desired_position,desired_orientation)
        finger_width = self.robot._get_finger_width()/2
        action = np.concatenate([action[:7], np.array([finger_width,finger_width])])
        return action




if __name__ == "__main__":


    env = MyRobotTaskEnv()
    observation, info = env.reset()

    for _ in range(1000):
        action = env.get_action()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

        time.sleep(0.01)


