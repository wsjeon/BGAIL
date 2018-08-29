# Reference
# 1) https://github.com/openai/imitation
# 2) https://github.com/andrewliao11/gail-tf
import h5py
import pickle as pkl


def load_dataset(filename):
    with h5py.File(filename, 'r') as f:
        dset_size = f['obs_B_T_Do'].shape[0]  # full dataset size

        exobs_B_T_Do = f['obs_B_T_Do'][:dset_size, ...][...]
        exa_B_T_Da = f['a_B_T_Da'][:dset_size, ...][...]
        exr_B_T = f['r_B_T'][:dset_size, ...][...]
        exlen_B = f['len_B'][:dset_size, ...][...]

    print('Expert dataset size: {} transitions ({} trajectories)'.format(exlen_B.sum(), len(exlen_B)))
    print('Expert average return (h5):', exr_B_T.sum(axis=1).mean())

    return exobs_B_T_Do, exa_B_T_Da, exr_B_T, exlen_B


if __name__ == '__main__':
    for task in ['cartpole', 'mountaincar', 'acrobot', 'reacher', 'hopper', 'walker', 'halfcheetah', 'ant', 'humanoid']:
        print('-'*50)
        print(task)
        print('-'*50)
        obs, acs, rews, ep_lens = load_dataset('./expert_trajs/trajs_{}.h5'.format(task))
        sample_trajs = []
        for i in range(obs.shape[0]):
            ob, ac, rew, ep_len = obs[i], acs[i], rews[i], ep_lens[i]
            traj = {"ob": ob, "ac": ac, "rew": rew, "ep_ret": rew.sum()}
            sample_trajs.append(traj)
    
        sample_ep_rets = [traj["ep_ret"] for traj in sample_trajs]
        print('Expert average return (pkl): {}'.format(sum(sample_ep_rets) / len(sample_ep_rets)))
        pkl.dump(sample_trajs, open("./expert_trajs/trajs_{}.pkl".format(task), "wb"))
