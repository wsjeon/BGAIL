from baselines.common.models import mlp


def mujoco():
    # TODO g_step, d_step? difference between defaults and args?
    return dict(
        policy_network=mlp(num_hidden=100, num_layers=2),
        classifier_network=mlp(num_hidden=100, num_layers=2),
        max_iters=2001,
        timesteps_per_batch=1000,
        max_kl=0.01,
        cg_iters=10,
        gamma=0.995,
        lam=0.97,
        entcoeff=0.0,
        cg_damping=0.1,
        vf_stepsize=1e-3,
        vf_iters=5,
        expert_trajs_path='./expert_trajs',
        ret_threshold=0.0,
        traj_limitation=500,
        g_step=1,
        d_step=5,
        classifier_entcoeff=1e-3,
        num_particles=5,
        d_stepsize=0.01,
        normalize_observations=True
    )
