import baselines.run as run
import tensorflow as tf
import numpy as np
import os.path as osp
import gym
from baselines.common.cmd_util import arg_parser, parse_unknown_args
from baselines.common.tf_util import get_session
from baselines.run import parse
from baselines import logger
try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def build_env(args):
    env_type, env_id = run.get_env_type(args.env)
    if env_type == 'mujoco':
        get_session(tf.ConfigProto(allow_soft_placement=True,
                                   intra_op_parallelism_threads=1,
                                   inter_op_parallelism_threads=1))

        env = gym.make(env_id)
        env.seed(args.seed)

    elif env_type == 'classic_control':
        env = gym.make(env_id)
        env.seed(args.seed)

    else:
        raise NotImplementedError

    return env


def common_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Walker2d-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='bgail')
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_path', help='Path to save trained model to', default='./outputs', type=str)

    return parser


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    alg_module = run.import_module('.'.join([alg, submodule]))

    return alg_module


def train(args, extra_args):
    env_type, env_id = run.get_env_type(args.env)
    # TODO: either 'mujoco' or 'classic_control'
    if env_type not in ['mujoco']:
        raise NotImplementedError

    seed = args.seed

    # TODO: either 'bgail' or 'gail'.
    if args.alg not in ['bgail']:
        raise NotImplementedError

    learn = run.get_learn_function(args.alg)
    alg_kwargs = run.get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = run.build_env(args)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(env=env, seed=args.seed, save_path=args.save_path, **alg_kwargs)


def main():
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = {k: parse(v) for k,v in parse_unknown_args(unknown_args).items()}


    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
    else:
        rank = MPI.COMM_WORLD.Get_rank()

    model, _ = train(args, extra_args)

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        model.save(save_path)


if __name__ == '__main__':
    run.build_env = build_env
    run.get_alg_module = get_alg_module
    run.train = train
    run.common_arg_parser = common_arg_parser
    run.main()
