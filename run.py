import baselines.run as run

def train(args, extra_args):
    env_type, env_id = run.get_env_type(args.env)
    # TODO: either 'mujoco' or 'classical_control'
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

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = run.get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,  
        seed=seed,
        **alg_kwargs
    )

    return model, env

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    alg_module = run.import_module('.'.join([alg, submodule]))
    return alg_module

if __name__ == '__main__':
    run.get_alg_module = get_alg_module
    run.train = train
    run.main()
