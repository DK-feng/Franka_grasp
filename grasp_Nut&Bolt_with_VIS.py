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
from skimage.filters import gaussian
import cv2


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
        self.jaw_size = 0.05

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

    def _vertical_move(self, distance:int, tight:bool=False):
        #仅仅竖直移动,单位m
        current_ee_position = self._get_ee_position()
        desired_ee_position = current_ee_position + np.array([0,0,distance])
        desired_ee_orientation = self._get_ee_orientation()
        action = self._inverse_kinematics(desired_ee_position,desired_ee_orientation)
        finger_width = self._get_finger_width()/2
        if tight:
            action = np.concatenate([action[:7], np.array([finger_width-0.01,finger_width-0.01])])
        else:
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
            fov=80.0,
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
    
    def get_rectangle(self, GGCNN_output):
        #处理GGCNN输出，得到用于可视化的数据
        index = torch.argmax(GGCNN_output[0]).cpu().numpy()
        GGCNN_output = [x.to('cpu') for x in GGCNN_output]
        cos2 = GGCNN_output[1].cpu().numpy().reshape(-1)[index]
        sin2 = GGCNN_output[2].cpu().numpy().reshape(-1)[index]
        width = GGCNN_output[3].cpu().numpy().reshape(-1)[index]
        row_index, column_index = index//300 +1, index%300 +1
        theta = np.arctan(sin2/cos2)/2  #是弧度并非角度

        return (column_index,row_index),theta,width,0.04

    def processing_ggcnnOutput(self,GGCNN_output,num_preds):
        #处理GGCNN的输出,将张量变成有用的数据
        sorted_pixel_map = GGCNN_output[0].flatten().sort(descending=True)
        indexes = sorted_pixel_map[1][:num_preds].cpu().numpy()
        actions = []
        desired_positions, desired_orientations = [], []
        central_points, thetas, widths = [],[],[]

        current_orientation_quaternion = self._get_ee_orientation()
        current_ee_position = self.sim.get_link_position(self.body_name,self.ee_link)
        current_orientation = p.getEulerFromQuaternion(self._get_ee_orientation())
        current_finger_width = self._get_finger_width()

        for index in indexes:
            cos2 = GGCNN_output[1].cpu().numpy().reshape(-1)[index]
            sin2 = GGCNN_output[2].cpu().numpy().reshape(-1)[index]
            width = GGCNN_output[3].cpu().numpy().reshape(-1)[index]
            row_index, column_index = index//300 +1, index%300 +1

            #根据抓取点在图像中不同的位置来不断微调desired_position,每次微调下降0.1cm
            desired_position = current_ee_position + 0.02*np.array([(column_index - 150)/300, (150 - row_index)/300, -0.05])

            theta = np.arctan(sin2/cos2)/2        #是弧度并非角度

            if current_orientation[2]-np.pi/2 < theta:
                desired_orientation = p.getQuaternionFromEuler(current_orientation + np.array([0 ,0 ,0.03])) 
            if current_orientation[2]-np.pi/2 > theta:
                desired_orientation = p.getQuaternionFromEuler(current_orientation + np.array([0 ,0 ,-0.03]))  


            desired_positions.append(desired_position)
            desired_orientations.append(desired_orientation)

            new_joints_state = self._inverse_kinematics(desired_position, desired_orientation)

            desired_finger_width = current_finger_width + 0.001 if current_finger_width-0.1 < width else current_finger_width - 0.001
            new_action = np.concatenate([new_joints_state[:7], np.array([desired_finger_width/2,
                                                                     desired_finger_width/2])])
            actions.append(new_action)
            central_point = np.array([column_index,row_index])
            central_points.append(central_point)


            thetas.append(theta)
            widths.append(width)

        all_loss = []
        for desired_position,desired_orientation in zip(desired_positions,desired_orientations):
            position_loss = np.sum((desired_position - current_ee_position)**2) * 10000
            orientation_loss = np.sum((desired_orientation - current_orientation_quaternion)**2) * 1000
            all_loss.append(position_loss+orientation_loss)
        index = np.array(all_loss).argmin()
        new_action = actions[index]
        vis_info = {'central_point': central_points[index],
                    'theta': thetas[index],
                    'width': widths[index]}

        return new_action, vis_info


class MyTask(Task):

    def __init__(self,sim):
        #在特定区域随机创建object和target
        super().__init__(sim)
        path = 'F:\Franka_grasp-main\obj_models\\Nut&Bolt.obj'
        mesh_scale = [7,7,7]
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=path,
            visualFramePosition=[0,0,0],
            meshScale=mesh_scale  
        )
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=path,
            collisionFramePosition=[0,0,0],
            meshScale=mesh_scale
        )
        self.objectId = p.createMultiBody(
            baseMass=1,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0.5, 0, 0.1],
            useMaximalCoordinates=True
        )
        self.sim.create_box(body_name='target',
                            half_extents=np.array([0.01,0.01,0.01]),
                            mass=0,
                            position=np.array([-0.5,-0.5,0.01]),
                            ghost=True,
                            rgba_color=np.array([255,97,0,80]))

    def reset(self):
        #随机生成物体和目标点 '''---------------------------------------------------------------------------
        object_x = np.random.uniform(low=0.3, high=0.4, size=1)
        object_y = np.random.uniform(low=-0.6, high=-0.5, size=1)
        object_position = np.concatenate((object_x, object_y, np.array([0.03])))
        object_orientation = np.random.uniform(low=-0.2,high=0.2,size=4)
        object_orientation += np.array([0.0, 0.86, 0.00, -0.5]) 
        target_y = np.random.uniform(low=0.4, high=0.6, size=1)
        target_position = np.concatenate((np.array([0.4]), target_y, np.array([0.01])))
        # target_orientation = np.random.uniform(low=-1,high=1,size=4)
        target_orientation = np.array([0,0,0,1])
        p.resetBasePositionAndOrientation(self.objectId,object_position,object_orientation)
        self.sim.set_base_pose(body='target',position=target_position,orientation=target_orientation)

    def get_obs(self):
        #返回物体的位置和目标点状态,前3位置后4姿态
        object_position = p.getBasePositionAndOrientation(self.objectId, 0)[0]
        object_orientation = p.getBasePositionAndOrientation(self.objectId, 0)[1]
        target_position = self.sim.get_base_position(body='target')
        target_orientation = self.sim.get_base_orientation(body='target')
        observation = np.concatenate((object_position,object_orientation,target_position,target_orientation))
        return observation  
    
    def get_achieved_goal(self):
        #返回物体目前的位置,前3位置后4姿态
        position = p.getBasePositionAndOrientation(self.objectId, 0)[0]
        orientation = p.getBasePositionAndOrientation(self.objectId, 0)[1]
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
        # loss = np.sum((achieved_goal-desired_goal)**2/2)
        # return np.array(loss<=0.1, dtype=bool)
        return False
    
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

    def __init__(self,render_mode='human',vis:bool=True):
        self.DEVICE = torch.device('cuda')
        sim = PyBullet(render_mode=render_mode)
        self.robot = MyRobot(sim)
        self.task = MyTask(sim)
        self.net = GGCNN().to(self.DEVICE)
        self.net.load_state_dict(torch.load(r"C:\Users/25630\Desktop/try\Franka_grasp\epoch_48_iou_0.88_statedict.pt"))  #加载训练好的权重
        self.vis = vis
        self.stage = 0
        self.counter = 0
        self.time = 0
        self.num_preds = 10
        super().__init__(self.robot,self.task)

    def get_action(self):
        #调整阶段
        self.time += 0.1
        # 40cm above the target
        if self.stage == 0.0:
            desired_position = self.task.get_obs()[:3] + np.array([0, 0, 0.4])
            self.current_position = self.robot._get_ee_position()
            self.position_change = desired_position - self.current_position
            self.stage = 0.1

        if self.stage == 0.1:
            self.counter += 1
            action = self.robot._inverse_kinematics(self.current_position + self.counter/300*self.position_change,
                                                    self.robot._get_ee_orientation())
            action = np.concatenate([action[:7], np.array([self.robot._get_finger_width()/2, 
                                                           self.robot._get_finger_width()/2])])
            if self.counter > 300:
                self.counter = 0
                self.stage = 1.0
            print('Moving......')
            return action

        # adjust grasp parameters
        if self.stage == 1.0:
            _,_,rgbImg,depthImg,_ = self.robot.setCameraAndGetPic()
            depthImg = np.clip(2*(depthImg - depthImg.mean()), -1, 1) 
            GGCNN_input = torch.tensor(depthImg).unsqueeze(0).unsqueeze(0).to(self.DEVICE)
            GGCNN_output = self.net(GGCNN_input)

            # if self.vis:
            #     center_pos, theta, opening, jaw_size = self.robot.get_rectangle(GGCNN_output)
            #     alpha = np.arctan(jaw_size/opening)
            #     half_cd =100 * np.sqrt(opening**2 + jaw_size**2) #Catercorner diagonal,对角线长度
            #     left_top = np.array([-half_cd*np.cos(alpha-theta), half_cd*np.sin(alpha-theta)]) + np.array(center_pos)
            #     left_bottom = np.array([-half_cd*np.cos(alpha+theta), -half_cd*np.sin(alpha+theta)]) + np.array(center_pos)
            #     right_top = np.array([half_cd*np.cos(alpha+theta), half_cd*np.sin(alpha+theta)]) + np.array(center_pos)
            #     right_bottom = np.array([half_cd*np.cos(alpha-theta), -half_cd*np.sin(alpha-theta)]) + np.array(center_pos)

            #     cv2.polylines(rgbImg, np.array([left_top,left_bottom,right_bottom,right_top],dtype=np.int32).reshape((-1,1,2)),
            #                    isClosed=True, color=(0,255,255), thickness=3)
            #     cv2.imshow('image',rgbImg)
            #     cv2.waitKey(10)
                
            action, vis_info = self.robot.processing_ggcnnOutput(GGCNN_output, num_preds=self.num_preds)
      
            if self.vis:
                center_pos = vis_info['central_point']
                theta = vis_info['theta']

                alpha = np.arctan(self.robot.jaw_size/vis_info['width'])
                half_cd =100 * np.sqrt(vis_info['width']**2 + self.robot.jaw_size**2) #Catercorner diagonal,对角线长度
                left_top = np.array([-half_cd*np.cos(alpha-theta), half_cd*np.sin(alpha-theta)]) + np.array(center_pos)
                left_bottom = np.array([-half_cd*np.cos(alpha+theta), -half_cd*np.sin(alpha+theta)]) + np.array(center_pos)
                right_top = np.array([half_cd*np.cos(alpha+theta), half_cd*np.sin(alpha+theta)]) + np.array(center_pos)
                right_bottom = np.array([half_cd*np.cos(alpha-theta), -half_cd*np.sin(alpha-theta)]) + np.array(center_pos)

                cv2.polylines(rgbImg, np.array([left_top,left_bottom,right_bottom,right_top],dtype=np.int32).reshape((-1,1,2)),
                               isClosed=True, color=(0,255,255), thickness=3)
                cv2.imshow('image',rgbImg)
                cv2.waitKey(10)

            ee_position = self.robot._get_ee_position()
            if ee_position[2] <= 0.20:       #只在摄像头高度大于20cm时微调,若小于这个距离,摄像头会失焦
                self.stage = 2.0  
                cv2.destroyAllWindows()  
            return action

        # other movement
        if self.stage == 2.0:
            action_1 = self.robot._vertical_move(-0.0003)     #下降
            if self.robot._get_ee_position()[2] < 0.03:
                self.stage = 3.0
                self.stage_2_num = 0
            print('Downward......')
            return action_1

        if self.stage == 3.0:
            self.stage_2_num += 1
            action_2 = self.robot._grip(-0.0005)
            if self.stage_2_num >= 300:
                self.stage = 4.0
                self.stage_3_num = 0
                self.finger_width = self.robot._get_finger_width()/2
                goal_info = self.task.get_goal()
                self.ee_info = self.robot.get_obs()
                self.position_change = goal_info[:3] + np.array([0,0,0.2]) - self.ee_info[:3]
            print('Grasping......')
            return action_2
        
        if self.stage == 4.0:
            self.stage_3_num += 1
            action = self.robot._inverse_kinematics(self.ee_info[:3] + self.stage_3_num/500*self.position_change,
                                                    self.ee_info[3:])
            action_3 = np.concatenate([action[:7], np.array([self.finger_width-0.01,self.finger_width-0.01])])
            print('Deliverying......')
            if self.stage_3_num > 500:
                self.stage = 5.0
            return action_3

        if self.stage == 5.0:
            action_4 = self.robot._vertical_move(-0.001,tight=True)
            if self.robot._get_ee_position()[2] < 0.04:
                self.stage = 6.0
                self.stage_5_num = 0
            print('Downward......')
            return action_4
        
        if self.stage == 6.0:
            self.stage_5_num += 1
            action_5 = self.robot._grip(0.0001)
            if self.stage_5_num > 200:
                self.stage = 7.0
            print('Loosening......')
            return action_5

        if self.stage == 7.0:
            action_6 = self.robot._vertical_move(0.001)    
            if self.robot._get_ee_position()[2] > 0.3:
                self.stage = 0.0
            print('Upward......')
            return action_6

    def _delivery(self):
        #送到目标点上空20cm并调整姿态
        total_timestep = 300
        finger_width = self.robot._get_finger_width()/2
        goal_info = self.task.get_goal()
        ee_info = self.robot.get_obs()
        position_change = goal_info[:3] + np.array([0,0,0.2]) - ee_info[:3]
        orientation_change = goal_info[3:] - ee_info[3:]
        for i in range(total_timestep):
            action = self.robot._inverse_kinematics(ee_info[:3] + i/total_timestep*position_change,
                                                    ee_info[3:] + i/total_timestep*orientation_change)
            action = np.concatenate([action[:7], np.array([finger_width,finger_width])])
            return action


if __name__ == "__main__":

    env = MyRobotTaskEnv()
    observation, info = env.reset()

    for _ in range(10000):
        action = env.get_action()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()

        
























