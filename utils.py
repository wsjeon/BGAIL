import tensorflow as tf
from gym.spaces import Discrete, Box
import os


def observation_placeholder(ob_space, batch_size=None, name='Ob'):
    '''
    Create placeholder to feed observations into of the size appropriate to the observation space

    Parameters:
    ----------

    ob_space: gym.Space     observation space

    batch_size: int         size of the batch to be fed into input. Can be left None in most cases.

    name: str               name of the placeholder

    Returns:
    -------

    tensorflow placeholder tensor
    '''

    assert isinstance(ob_space, Discrete) or isinstance(ob_space, Box), \
        'Can only deal with Discrete and Box observation spaces for now'
    if isinstance(ob_space, Box):
        return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=tf.float32, name=name)
    else:
        try:
            return tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
        except AttributeError:
            return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=tf.int32, name=name)


def logsigmoid(a):
    return - tf.nn.softplus(-a)  # Equivalent to tf.log(tf.sigmoid(a))


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent


def load_state(fname, saver):
    saver.restore(tf.get_default_session(), fname)


def save_state(fname, saver, global_step=None):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    saver.save(tf.get_default_session(), fname, global_step=global_step, write_meta_graph=False)


def FileWriter(save_path):
    os.makedirs(save_path, exist_ok=True)
    return tf.summary.FileWriter(save_path)
