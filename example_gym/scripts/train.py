import numpy as np
import os
from datetime import datetime

import isaacgym
from example_gym.envs import *
from example_gym.utils import task_registry
from example_gym.utils.helpers import get_args 

def train(args):
    env, env_cfg = task_registry.make_env(name = args.task,args=args)
    ppo_runner,train_cfg = task_registry.make_alg_runner(env=env,name=args.task,args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg)

if __name__ == "__main__":
    args = get_args()
    train(args)