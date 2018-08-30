from baselines.common.models import mlp


def classic_control():
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
        num_expert_trajs=25,
        g_step=1,
        d_step=5,
        classifier_entcoeff=1e-3,
        num_particles=5,
        d_stepsize=0.01,
        normalize_observations=True,
        observation_dependent_var=False,
        use_classifier_logsumexp=True,
        use_reward_logsumexp=False,
        use_svgd=True
    )


def mujoco():
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
        num_expert_trajs=25,
        g_step=1,
        d_step=5,
        classifier_entcoeff=1e-3,
        num_particles=5,
        d_stepsize=0.01,
        normalize_observations=True,
        observation_dependent_var=True,
        use_classifier_logsumexp=True,
        use_reward_logsumexp=False,
        use_svgd=True
    )


def classic_control_gail():
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
        num_expert_trajs=25,
        g_step=1,
        d_step=5,
        classifier_entcoeff=1e-3,
        num_particles=1,
        d_stepsize=0.01,
        normalize_observations=True,
        observation_dependent_var=False,
        use_classifier_logsumexp=False,
        use_reward_logsumexp=False,
        use_svgd=False
    )


def mujoco_gail():
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
        num_expert_trajs=25,
        g_step=1,
        d_step=5,
        classifier_entcoeff=1e-3,
        num_particles=1,
        d_stepsize=0.01,
        normalize_observations=True,
        observation_dependent_var=True,
        use_classifier_logsumexp=False,
        use_reward_logsumexp=False,
        use_svgd=False
    )
