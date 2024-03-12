import os

TASK_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TASK_GYM_ENVS_DIR = os.path.join(TASK_GYM_ROOT_DIR, 'example_gym','envs')
TASK_ASSET_DIR = os.path.join(TASK_GYM_ROOT_DIR,'resources')