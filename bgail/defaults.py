from baselines.common.models import mlp


def mujoco():
    # TODO g_step, d_step? difference between defaults and args?
    return dict(
        network = mlp(num_hidden=100, num_layers=2),
        timesteps_per_batch=1000,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.995,
        lam=0.97,
        entcoeff=0.0,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
        g_step=1,
        d_step=1,
        ret_threshold=0.0,
        traj_limitation=500,
        adversary_entcoeff=1e-3,
        max_iters=2001,
        num_particles=5,
        d_stepsize=0.01,
        save_path='./outputs'
    )
