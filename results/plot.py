from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import itertools
import os
import numpy as np
import sys; sys.path.insert(0, '..')


def plot(load_path):

    alg = ['bgail', 'gail']  # 1
    env = ['Hopper-v1', 'Walker2d-v1', 'HalfCheetah-v1']  # 1
    num_expert_trajs = [25]  # 1
    d_step = [5, 10]  # 2
    num_particles = [1, 5, 9]  # 3
    timesteps_per_batch = [1000]  # 1
    seed = list(range(5))  # 5  --->   900 Processes in total

    hyperparameters_list = list(itertools.product(alg, env,
                                                  use_classifier_logsumexp,
                                                  num_expert_trajs, d_step, num_particles,
                                                  timesteps_per_batch))

    num_colors = int(len(hyperparameters_list) / len(env) / len(alg))
    cm = plt.get_cmap('brg')
    colors = [cm(i/num_colors) for i in range(num_colors)] * len(env) * len(alg)

    for hyperparameters in hyperparameters_list:
        additional_path = os.path.join(*[str(h) for h in hyperparameters])
        path = os.path.join(load_path, additional_path)
        if os.path.exists(path):
            pass
        else:
            continue

        datas = []

        print('Load event file from path {}'.format(path))

        for s in seed:
            spath = os.path.join(path, str(s), 'logs')
            if os.path.exists(spath):
                event = EventAccumulator(spath)
                event.Reload()
                _, _, values = zip(*event.Scalars('summary/average_return.scalar.summary_1'))

                datas.append(values)

                print('\t {}'.format('='*40))
                print('\t o Seed = {}'.format(s))
                print('\t o Algorithm: {}'.format(hyperparameters[0].upper()))
                print('\t o Episode lengths = {}'.format(len(values)))
                print('\t o Score = {:.3f}'.format(values[-1]))

        min_len = min([len(v) for v in datas])
        datas = [v[:min_len] for v in datas]
        m = np.mean(datas, axis=0)
        s = np.std(datas, axis=0)
        c = 1.96 * s / len(datas)

        alg_id = alg.index(hyperparameters[0])
        env_id = env.index(hyperparameters[1])
        fig_id = alg_id * len(env) + env_id
        plt.figure(fig_id)
        hyperparameter_id = hyperparameters_list.index(hyperparameters)
        color = colors[hyperparameter_id]
        window_size = 10
        window = np.ones(window_size) / window_size
        ma = np.convolve(m, window, mode='same')[:m.shape[0]]

        plt.plot(np.arange(len(m)), ma, color=color, label=path, linewidth=0.8, zorder=2)
        plt.fill_between(np.arange(len(m)), m-c, m+c, facecolor=color, alpha=0.2, edgecolor=None, zorder=1)
        plt.legend()

    for alg_id in range(len(alg)):
        for env_id in range(len(env)):
            fig_id = alg_id * len(env) * env_id
            plt.figure(fig_id)
            plt.show()


if __name__ == '__main__':
    load_path = '../outputs'
    plot(load_path)