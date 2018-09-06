import baselines.run as run
import gym
from baselines.common.cmd_util import arg_parser, parse_unknown_args


def build_env(args):
    env_type, env_id = run.get_env_type(args.env)
    if env_type in ['mujoco', 'classic_control']:
        env = gym.make(env_id)
        env.seed(args.seed)
    else:
        raise NotImplementedError

    return env


def common_arg_parser():
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='bgail')
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', type=float, default=1.0)
    parser.add_argument('--save_path', help='Path to save trained model to', type=str, default='./outputs')
    parser.add_argument('--load_path', help='Path to load trained model for evaluation', type=str, default=None)
    parser.add_argument('--render', help='Whether to display the simulation or not', default=False)

    return parser


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    alg_module = run.import_module('.'.join([alg, submodule]))

    return alg_module


def train(args, extra_args):
    env_type, env_id = run.get_env_type(args.env)

    if args.alg == 'gail':
        env_type += '_gail'
        args.alg = 'bgail'
    elif args.alg not in ['bgail', 'gail']:
        raise NotImplementedError

    learn = run.get_learn_function(args.alg)
    alg_kwargs = run.get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(env=env,
                  seed=args.seed,
                  save_path=args.save_path,
                  load_path=args.load_path,
                  render=args.render,
                  **alg_kwargs)


def main():
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args()
    extra_args = {k: run.parse(v) for k,v in parse_unknown_args(unknown_args).items()}

    train(args, extra_args)


if __name__ == '__main__':
    run.get_alg_module = get_alg_module
    main()
