import tensorflow as tf
from gym.spaces import Discrete, Box


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
    dtype = tf.int32 if isinstance(ob_space, Discrete) else tf.float32

    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)


def logsigmoid(a):
    return - tf.nn.softplus(-a)  # Equivalent to tf.log(tf.sigmoid(a))


def logit_bernoulli_entropy(logits):
    ent = (1. - tf.nn.sigmoid(logits)) * logits - logsigmoid(logits)
    return ent
