from gailtf.baselines.common.mpi_running_mean_std import RunningMeanStd
import tensorflow as tf
from gailtf.baselines.common import tf_util as U
from gailtf.common.tf_util import *
import numpy as np
from gym import spaces
from tensorflow.contrib.layers import fully_connected as fc

##
# import tensorflow as tf
from baselines.common import tf_util
from baselines.a2c.utils import fc
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_placeholder, encode_observation
from baselines.common.tf_util import adjust_shape
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.models import get_network_builder
from baselines.common.policies import _normalize_clip_observation

import gym

class TransitionClassifier(object):
    """
    Encapsulates fields and methods for discriminator
    """

    def __init__(self, env, observations, actions, logits_list, **tensors):
        """
        Parameters:
        ----------
        env             RL environment

        observations    tensorflow placeholder in which the observations will be fed

        latent          latent state from which policy distribution parameters should be inferred

        vf_latent       latent state from which value function should be inferred (if None, then latent is used)

        sess            tensorflow session to run calculations in (if None, default session is used)

        **tensors       tensorflow tensors for additional attributes such as state or mask

        """

        self.X = observations
        self.A = actions
        self.logits_list = logits_list
        self.__dict__.update(tensors)
        self.sess = tf.get_default_session()

    def build_graph(self):


    def _evaluate(self, variables, observation, **extra_feed):
        feed_dict = {self.X: adjust_shape(self.X, observation)}
        for inpt_name, data in extra_feed.items():
            if inpt_name in self.__dict__.keys():
                inpt = self.__dict__[inpt_name]
                if isinstance(inpt, tf.Tensor) and inpt._op.type == 'Placeholder':
                    feed_dict[inpt] = adjust_shape(inpt, data)

        return sess.run(variables, feed_dict)

    def value(self, ob, *args, **kwargs):
        """
        Compute value estimate(s) given the observation(s)

        Parameters:
        ----------

        observation     observation data (either single or a batch)

        **extra_feed    additional data such as state or mask (names of the arguments should match the ones in constructor, see __init__)

        Returns:
        -------
        value estimate
        """
        return self._evaluate(self.vf, ob, *args, **kwargs)

    def save(self, save_path):
        tf_util.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        tf_util.load_state(load_path, sess=self.sess)

def build_classifier(env, classifier_network, num_particles, normalize_observations=True):
    if isinstance(classifier_network, str):
        raise NotImplementedError

    def classifier_fn():
        ob_space, ac_space = env.observation_space, env.action_space
        Xs, As, Ls = {}, {}
        for c in ['a', 'e']:
            Xs[c] = observation_placeholder(ob_space, name='Ob_'+c)
            As[c] = observation_placeholder(ac_space, name='Ac_'+c)
            Ls[c] = tf.placeholder()
        if not isinstance(ob_space, spaces.Box):
            raise NotImplementedError

        if isinstance(ac_space, spaces.Box):
            if normalize_observations and X.dtype == tf.float32 and A.dtype == tf.float32:
                encoded_x, ob_rms = _normalize_clip_observation(X)
                encoded_a, ac_rms = _normalize_action(A)
                extra_tensors['ob_rms'], extra_tensors['ac_rms'] = ob_rms, ac_rms
            else:
                encoded_x, encoded_a = X, A

            logits_list = []
            for i in range(num_particles):
                with tf.variable_scope('classifier{}'.format(i), reuse=tf.AUTO_REUSE):
                    flatten_x, flatten_a = tf.layers.flatten(encoded_x), tf.layers.flatten(encoded_a)
                    concat = tf.concat([flatten_x, flatten_a], axis=1)
                    classifier_latent, recurrent_tensors = classifier_network(concat)
                    if recurrent_tensors is not None:
                        raise NotImplementedError
                    logits = fc(classifier_latent, 'out', nh=1, init_scale=np.sqrt(2))
                    logits_list.append(logits)

            classifier = TransitionClassifier(
                env=env,
                observations=Xs,
                actions=As,
                num_particles
                )
        elif isinstance(ac_space, spaces.Discrete):
            raise NotImplementedError
        else:
            raise NotImplementedError

        return classifier

    return classifier_fn



##







class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary", use_surrogate=True, env_id='Hopper-v1'):
        self.scope = scope
        self.hidden_size = hidden_size
        self.ob_space = env.observation_space
        self.ac_space = env.action_space
        self.observation_shape = env.observation_space.shape
        if isinstance(env.action_space, spaces.Box):
            assert len(env.action_space.shape) == 1
            self.actions_shape = env.action_space.shape
        elif isinstance(env.action_space, spaces.Discrete):
            self.actions_shape = (1,)
        else:
            raise NotImplementedError
        self.build_ph()
        # Build graph
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        # [?, ?, 0,    ?, 0, 0].T
        generator_o_a = tf.reduce_sum(
                tf.concat([self.generator_obs_ph, self.generator_acs_ph], axis=1), axis=1)

        trues = tf.cast(tf.ones_like(generator_o_a), dtype=tf.bool)
        falses = tf.cast(tf.zeros_like(generator_o_a), dtype=tf.bool)
        mask = tf.where(generator_o_a != 0.0, trues, falses)

        generator_loss = tf.boolean_mask(generator_loss, mask) # [logD, logD, 0,   logD, 0, 0].T
        generator_loss = tf.reduce_sum(
                tf.reshape(generator_loss,
                    tf.convert_to_tensor([self.generator_n_epi_ph, -1])),
                axis=1) # [F_A^(1), F_A^(2), ... ]

        self.generator_loss = tf.reduce_logsumexp(generator_loss)

        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_o_a = tf.reduce_sum(
                tf.concat([self.expert_obs_ph, self.expert_acs_ph], axis=1), axis=1)

        trues = tf.cast(tf.ones_like(expert_o_a), dtype=tf.bool)
        falses = tf.cast(tf.zeros_like(expert_o_a), dtype=tf.bool)
        mask = tf.where(expert_o_a != 0.0, trues, falses)

        expert_loss = tf.boolean_mask(expert_loss, mask)
        expert_loss = tf.reduce_sum(
                tf.reshape(expert_loss,
                    tf.convert_to_tensor([self.expert_n_epi_ph, -1])),
                axis=1)

        self.expert_loss = tf.reduce_logsumexp(expert_loss)

        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        self.entropy_loss = - entcoeff * entropy

        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))

        # Losses
        self.losses = [self.generator_loss, self.expert_loss, entropy, self.entropy_loss, generator_acc, expert_acc]
        self.loss_names = ["generator_loss", "expert_loss"] #!! right now, I just consider this two losses!
        self.total_loss = self.generator_loss + self.expert_loss + self.entropy_loss
        # Build Reward for policy
        # Settings below give the positive rewards!
        if env_id in ['MountainCar-v0']:
            self.reward_op = tf.log(tf.nn.sigmoid(generator_logits)+1e-8)
        else:
            self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        self.var_list = self.get_trainable_variables()
        self.generator_lossandgrad =\
                U.function([self.generator_obs_ph, self.generator_acs_ph, self.generator_n_epi_ph],
                        [self.generator_loss, U.flatgrad(self.generator_loss, self.var_list)])
        self.expert_lossandgrad =\
                U.function([self.expert_obs_ph, self.expert_acs_ph, self.expert_n_epi_ph],
                        [self.expert_loss, U.flatgrad(self.expert_loss, self.var_list)])
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                self.losses + [U.flatgrad(self.total_loss, self.var_list)])

        self.get_flat = U.GetFlat(self.get_trainable_variables())
        self.set_from_flat = U.SetFromFlat(self.get_trainable_variables())

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.generator_n_epi_ph = tf.placeholder(tf.int32)
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")
        self.expert_n_epi_ph = tf.placeholder(tf.int32)

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            if isinstance(self.ac_space, spaces.Box): # Continuous action space
                _input = tf.concat([obs, acs_ph], axis=1) # concatenate the two input
                p_h1 = fc(_input, self.hidden_size, activation_fn=tf.nn.tanh)
                p_h2 = fc(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
                logits = fc(p_h2, 1, activation_fn=tf.identity)
            elif isinstance(self.ac_space, spaces.Discrete): # Discrete action space
                _input = obs
                p_h1 = fc(_input, self.hidden_size, activation_fn=tf.nn.tanh)
                p_h2 = fc(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
                params = fc(p_h2, self.ac_space.n, activation_fn=tf.identity)
                subindices = tf.reshape(tf.range(tf.shape(acs_ph)[0]), [-1, 1])
                acs = tf.cast(acs_ph, dtype=tf.int32)
                indices = tf.concat([subindices, acs], axis=1)
                logits = tf.gather_nd(params, indices)
            else:
                raise NotImplementedError
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = U.get_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0).reshape(-1, 1)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0).reshape(-1, 1)
        feed_dict = {self.generator_obs_ph:obs, self.generator_acs_ph:acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward

def _normalize_action(a):
    rms = RunningMeanStd(shape=a.shape[1:])
    norm_a = (a - rms.mean) / rms.std
    return norm_a, rms
