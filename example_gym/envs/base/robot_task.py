from example_gym import TASK_GYM_ROOT_DIR
from time import time
from warnings import WarningMessage
import numpy as np
import os
import torch

from .base_task import BaseTask
from .robot_config import LeggedRobotCfg
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



    def step(self,actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """



    def post_physics_step(self):