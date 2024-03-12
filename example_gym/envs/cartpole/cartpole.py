import numpy as np
import os
from example_gym import TASK_ASSET_DIR
from isaacgym import gymapi
from isaacgym import gymutil

gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()

sim_params.dt = 1/60
sim_params.gravity = gymapi.Vec3(0.0,0.0,-9.8)
sim_params.up_axis = gymapi.UP_AXIS_Z

device, device_id = gymutil.parse_device_str('cuda')

asset_root = os.path.join(TASK_ASSET_DIR,'example')
asset_file = os.path.basename(os.path.join(TASK_ASSET_DIR,'example','cartpole.urdf'))

sim = gym.create_sim()
robot_asset = gym.load_asset(sim,asset_root,asset_file)

num_envs = 20
envs_per_row = 5
env_spacing = 5
env_lower = gymapi.Vec3(-env_spacing,0.0,env_spacing)
env_upper = gymapi.Vec3(env_spacing,env_spacing,env_spacing)


cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)


for i in range(num_envs):
    env = gym.create_env(sim,env_lower,env_upper,envs_per_row)

    pose = gymapi.Transform(gymapi.Vec3(0.0, 2.0, 0.0), gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107))
    gym.create_actor(env,robot_asset,pose,'cart_pole')


while not gym.query_viewer_has_closed(viewer):
    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)