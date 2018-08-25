from gym import spaces

class TransitionClassifier(object):
    def __init__(self, env, num_particles, normalize_observations=True):
        self.env = env
        self.num_particles = num_particles

    def make_placeholder(self):
        # TODO: make placeholder Xs, As, Ls (Ls for varying length of episode)
        pass

    def normalize(self):
        # TODO: normalize state with _normalize_clip_observation
        # TODO: normalize action (when continuous) with _normalize_action (without clip!!!)
        pass

    def build_graph(self):
        # TODO: build graph with multiple classifiers
        # TODO: make logits
        # TODO: Remind networks are shared for agent and expert.
        pass

    def build_loss(self):
        # TODO: Define loss and make gradients and vars HERE!!
        pass

    def get_grads_and_vars(self):
        # TODO: for SVGD
        # TODO: Not define gradients and vars here!
        pass

    def get_rewards(self):
        # TODO: to give rewards for policy learning
        # TODO: Not define rewards here!
        pass

def build_classifier(env, classifier_network, num_particles, normalize_observations=True):
    if isinstance(classifier_network, str):
        raise NotImplementedError

    def classifier_fn():
        ob_space, ac_space = env.observation_space, env.action_space

        if not isinstance(ob_space, spaces.Box):
            raise NotImplementedError

        if isinstance(ac_space, spaces.Box):
            classifier = TransitionClassifier(
                env=env,
                num_particles=num_particles,
                normalize_observations=normalize_observations
                )
        elif isinstance(ac_space, spaces.Discrete):
            raise NotImplementedError
        else:
            raise NotImplementedError

        return classifier

    return classifier_fn()
