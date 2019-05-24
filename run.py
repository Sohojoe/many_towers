from sohojoe_wrappers import RenderObservations
from obstacle_tower_env import ObstacleTowerEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv

def otc_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training processes to use')
    parser.add_argument(
        '--port-offset',
        type=int,
        default=0,
        help='which port number to start with')
    parser.add_argument(
        '--pause',
        type=int,
        default=1,
        help='how many seconds to pause between creating environments')
    parser.add_argument(
        'environment_filename',
        default='./ObstacleTower/obstacletower',
        nargs='?')
    
    return parser

def make_env_all_params(rank, args):
    from time import sleep
    sleep_time = rank
    sleep_multiple = args.pause
    sleep(sleep_multiple * sleep_time)

    show_obs = rank == 0

    rank = args.port_offset + rank

    # handle port clashes
    if rank >= 35:
        rank += 1
    
    environment_path = args.environment_filename
    
    env = ObstacleTowerEnv(
        environment_path,
        worker_id=rank,
        timeout_wait=6000,
        retro=True,
        realtime_mode=False)
    if show_obs:
        env = RenderObservations(env)
    return env

def make_otc_env(args):
    def make_env(rank):
        def _thunk():
            env = make_env_all_params(rank, args)
            return env

        return _thunk

    env_fns = [
        make_env(i)
        for i in range(args.num_processes)
    ]

    if args.num_processes == 1:
        envs = DummyVecEnv(env_fns)
    else:
        envs = ShmemVecEnv(env_fns)

    return envs


def main():
    parser = otc_arg_parser()
    args = parser.parse_args()
    envs = make_otc_env(args)

    print ('---- created:', args.num_processes, 'environments')

    while (True):
        actions = [envs.action_space.sample() for i in range(args.num_processes)]
        envs.step_async(actions)
        obs, reward, done, info = envs.step_wait()


if __name__ == "__main__":
    main()