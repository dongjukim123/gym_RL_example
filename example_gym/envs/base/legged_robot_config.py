from .base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4000
        num_observations = 235





    class terrain:
        mesh_type = 'trimesh' # 'heightfield' #none, plane, heightfield or trimesh
        horizontal_scale = 0.1 #[m]
        vertical_sacle = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        #rough terrain only:
        measure_heights = True

    class commands:
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s] 이 값이 클수록 lin_vel_x, lin_vel_y, ang_vel_yaw, heading 값이 자주 바뀜
        heading_command = True # if True: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control:
        control_type = 'p' # p : POSTION , V:velocity, T:Torques
        #PD drive parameters
        stiffness = {'joint_a':15.0,'joint_b':10.} #[N*m/rad]
        damping = {'joint_a':1.0,'joint_b':1.5} #[N*m*s/rad]
        #action scale : target_angle = actionsScale*action + defaultangle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file =""
        name = "legged_robot"
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors

    class init_state:
        pos = [0.0,0.0,1.] # x y z (m)
        rot = [0.0,0.0,0.0,1.0] # x y z w [quat]
        lin_vel = [0.0,0.0,0.0] # x y z [m/s]
        ang_vel = [0.0,0.0,0.0] # x y z [rad/x]
        default_joint_angles ={  #target angles when action = 0.0
            "joint_a": 0.,
            "joint_b": 0.}

    class normalization:
        class obs_scales:
            lin_vel = 2.0

        clip_actions = 100.
        clip_observations = 100.

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15



class LeggedRobotCfgPPO(BaseConfig):