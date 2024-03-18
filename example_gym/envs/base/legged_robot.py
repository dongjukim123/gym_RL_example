from example_gym import TASK_GYM_ROOT_DIR
from time import time
from warnings import WarningMessage
import numpy as np
import os
import torch

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

from .base_task import BaseTask
from .legged_robot_config import LeggedRobotCfg


from legged_gym.utils.terrain import Terrain
class LeggedRobot(BaseTask):
    def __init__(self,cfg:LeggedRobotCfg,sim_params, physics_engine, sim_device, headless ):

        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params

        super().__init__(self.cfg,sim_params,physics_engine,sim_device,headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos,self.cfg.viewer.lookat)
        self._init_buffers()

    def step(self,actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # action 클립 (범위를 제한 해준다)
        # self.device 는 cuda or cpu 두 개 중에 한 개
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions,-clip_actions,clip_actions).to(self.device)
        #step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim,gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device =='cpu':
                self.gym.fetch_results(self.sim,True)
            # update dof state buffer
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        # update acotr root state buffer
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # Updates buffer state for net contact force tensor
        self.gym.refresh_net_contact_force_tensor(self.sim)
        
        self.episode_length_buf +=1
        self.common_step_counter +=1

        #prepare quantities
        self.base_quat[:] = self.root_states[:,3:7]

    def set_camera(self, position, lookat):



    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """

        # get gym Gpu state tensors
        """ acquire_actor_root_state_tensor : shpaes(num_actors, 13), position([0:3]), rotation([3:7]), 
                                              linear_velocity([7:10]), angular_velocity([10:13])
                                              return type: gymapi.tensor
            acquire_dof_state_tensor : shape(num_dofs,2) 2 : num_dof = num_actors * each dof of robot,  position, velocity
            
            acquire_net_contact_force_tensor : shape(num_rigid_bodies,3) :contact force x,y,z axis
        """
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        #create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state) # torch.shape(num_actors(액터의 개수),12)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor) # torch.shape(num_dof,2) --> num_dof = actor 개수(envs 개수) * dof of robot
        self.dof_pos = self.dof_state.view(self.num_envs,self.num_dof,2)[...,0]
        self.dof_vel = self.dof_state.view(self.num_envs,self.num_dof,2)[...,1]
        self.base_quat = self.root_states[:,3:7] # rotation x y z w quaternion


        self.base_quat = self.root_states[:, 3:7]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        # initialize some data used later on
        self.common_step_counter =0

        self.p_gains = torch.zeros(self.num_actions,dtype=torch.float,device=self.device,requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions,dtype=torch.float,device=self.device,requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1.,self.up_axis_idx),device=self.device).repeat((self.num_envs,1))



        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof,dtype=torch.float,device=self.device,requires_grad=False)
        for i in range(self.num_dof):
            name = self.dof_names[i] # name : urdf상의 조인트 이름
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
                if not found:
                    self.p_gains[i] = 0.
                    self.d_gains[i] = 0.
                    if self.cfg.control.control_type in ["P","V"]:
                        print(f"PD gain of joint {name} were not defined, setting them to zero")
        
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0) # torch.shape(1,12) 나중에 연산을 수행하기 위해 차원을 추가하였음



    def _prepare_reward_function(self):
        

    def _compute_torques(self,actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #PD controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        return torch.clip(torques,-self.torque_limits,self.torque_limits)
    


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(TASK_GYM_ROOT_DIR=TASK_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path) # dirname of urdf
        asset_file = os.path.basename(asset_path) # filename of urdf

        #gymapi.AssetOptions.
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity        

        robot_asset = self.gym.load_asset(self.sim,asset_root,asset_file,asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

    def create_sim(self):
        """ creates simulation, terrain and environments
        """
        
        self.up_axis_idx = 2 # 2 for z, 1 for y --> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type

        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain,self.num_envs)
        if mesh_type =='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()



    def _create_ground_plane(self):



    def _create_heightfield(self):


    def _create_trimesh(self):
