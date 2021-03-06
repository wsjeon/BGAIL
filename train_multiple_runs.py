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

    parser.add_argument('--use_classifier_logsumexp', help='Use classifier logsumexp or not', default=True)

    return parser


def main():
    argparser = arg_parser_of_interest()
    args, _ = argparser.parse_known_args()

    alg = ['bgail', 'gail']  # 2
    env = ['Walker2d-v1', 'HalfCheetah-v1', 'Hopper-v1', 'Ant-v1', 'Humanoid-v1']  # 5
    use_classifier_logsumexp = [True, False] # 2
    num_expert_trajs = [25]  # 1
    d_step = [5]  # 1
    num_particles = [1, 5, 9]  # 3
    timesteps_per_batch = [1000]  # 1
    seed = list(range(5))  # 5  --->   300 Processes in total

    max_iters = 4001

    hyperparameters_list = list(itertools.product(alg, env,
                                                  use_classifier_logsumexp,
                                                  num_expert_trajs, d_step, num_particles,
                                                  timesteps_per_batch, seed))
    hyperparameters = list(hyperparameters_list[args.process_id])
    args.alg, args.env, \
    args.use_classifier_logsumexp, \
    args.num_expert_trajs, args.d_step, args.num_particles, args.timesteps_per_batch, args.seed \
        = hyperparameters

    if args.alg == 'gail':
        hyperparameters[2] = args.use_classifier_logsumexp = False
    if args.env == 'Humanoid-v1':
        hyperparameters[3] = args.num_expert_trajs = 240
        max_iters = 15001
    elif args.env == 'Ant-v1':
        max_iters = 10001

    additional_path = os.path.join(*[str(h) for h in hyperparameters])
    args.save_path = os.path.join(args.save_path, additional_path)

    # FILTERING: if some condition is satisfied, do not run.
    if os.path.exists(args.save_path):
        assert False

    interpreter = '/home/wsjeon/anaconda3/envs/bgail/bin/python '
    command = interpreter + 'run.py' + ' --max_iters={}'.format(str(max_iters))
    for key in ['alg', 'env',
                'use_classifier_logsumexp',
                'num_expert_trajs', 'd_step', 'num_particles', 'timesteps_per_batch', 'seed', 'save_path']:
        command += ' --' + key + '={}'.format(str(args.__dict__[key]))

    call(command, shell=True, executable='/bin/bash')


if __name__ == '__main__':
    main()
