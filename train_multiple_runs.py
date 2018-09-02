from baselines.common.cmd_util import arg_parser
import itertools
import os
from subprocess import call


def arg_parser_of_interest():
    parser = arg_parser()
    parser.add_argument('--process_id', help='Process ID (among all hyperparameter combinations)', type=int, default=0)

    parser.add_argument('--alg', help='Algorithm', type=str, default='bgail')
    parser.add_argument('--env', help='environment ID', type=str, default='Hopper-v1')
    parser.add_argument('--num_expert_trajs', help='Number of expert trajectories for training', default=25, type=int)
    parser.add_argument('--d_step', help='Number of classifier update steps for each iteration', default=5, type=int)
    parser.add_argument('--num_particles', help='Number of SVGD or Ensemble classifiers', default=5, type=int)
    parser.add_argument('--timesteps_per_batch', help='Minimum batch size for each iteration', default=1000, type=int)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument('--save_path', help='Path to save trained model to', default='./outputs', type=str)

    return parser


def main():
    argparser = arg_parser_of_interest()
    args, _ = argparser.parse_known_args()

    alg = ['bgail', 'gail']  # 2
    env = ['Hopper-v1', 'Walker2d-v1', 'HalfCheetah-v1', 'Ant-v1', 'Humanoid-v1']  # 5
    num_expert_trajs = [25]  # 1
    d_step = [5, 10]  # 2
    num_particles = [1, 5, 10]  # 3
    timesteps_per_batch = [1000]  # 1
    seed = list(range(15))  # 15  --->   900 Processes in total

    hyperparameters_list = list(itertools.product(alg, env, num_expert_trajs, d_step, num_particles,
                                                  timesteps_per_batch, seed))
    hyperparameters = list(hyperparameters_list[args.process_id])
    args.alg, args.env, args.num_expert_trajs, args.d_step, args.num_particles, args.timesteps_per_batch, args.seed \
        = hyperparameters

    # Filtering
    if args.env in ['Ant-v1', 'Humanoid-v1']:
        assert False
    if args.alg == 'bgail':
        assert False 

    if args.env == 'Humanoid-v1':
        args.num_expert_trajs = 240
        args.timesteps_per_batch = 5000
        hyperparameters[2] = args.num_expert_trajs
        hyperparameters[5] = args.timesteps_per_batch
    elif args.env == 'Ant-v1':
        args.timesteps_per_batch = 5000
        hyperparameters[5] = args.timesteps_per_batch

    additional_path = os.path.join(*[str(h) for h in hyperparameters])
    args.save_path = os.path.join(args.save_path, additional_path)

    interpreter = '/home/wsjeon/anaconda3/envs/bgail/bin/python '
    command = interpreter + 'run.py'
    for key in ['alg', 'env', 'num_expert_trajs', 'd_step', 'num_particles', 'timesteps_per_batch', 'seed', 'save_path']:
        command += ' --' + key + '={}'.format(str(args.__dict__[key]))

    call(command, shell=True, executable='/bin/bash')


if __name__ == '__main__':
    main()
