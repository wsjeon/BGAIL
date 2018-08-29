import tensorflow as tf
import baselines.common.distributions as distributions
from baselines.a2c.utils import fc


class DiagGaussianPdType(distributions.PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return distributions.DiagGaussianPd

    def pdfromlatent(self, latent_vector, init_scale=1.0, init_bias=0.0, observation_dependent_var=False):
        if not observation_dependent_var:
            mean = fc(latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias)
            logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = fc(latent_vector, 'pi', self.size * 2, init_scale=init_scale, init_bias=init_bias)
            mean, logstd = pdparam[:, :self.size], pdparam[:, self.size:]
        return self.pdfromflat(pdparam), mean

    def param_shape(self):
        return [2*self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


make_pdtype = distributions.make_pdtype
distributions.DiagGaussianPdType = DiagGaussianPdType
