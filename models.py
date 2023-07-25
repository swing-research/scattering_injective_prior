import tensorflow as tf
from utils import *
import glow_ops as g
import tensorflow_probability as tfp

tfb = tfp.bijectors
tfd = tfp.distributions


class prior(tf.keras.Model):
    """Defines the low dimensional distribution as Gaussian"""
    def __init__(self, **kwargs):
        super(prior, self).__init__()

        latent_dim = kwargs.get('latent_dim', 64)
        self.mu = tf.Variable(tf.zeros(latent_dim),
                              dtype=tf.float32, trainable=True)
        self.logsigma = tf.Variable(tf.zeros(latent_dim),
                                    dtype=tf.float32, trainable=True)
        self.prior = tfd.MultivariateNormalDiag(
            self.mu, tf.math.exp(self.logsigma))


class bijective(tf.keras.Model):
    def __init__(self, **kwargs):
        super(bijective, self).__init__()
        """ Bijective subnetwork"""

        self.depth = kwargs.get('revnet_depth', 3) # revnet depth

        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(coupling_type='affine', depth = self.depth , latent_model = True) 
        for _ in range(6)]

    def call(self, x, reverse=False , training = True):

        x = tf.reshape(x, [-1, 4, 4, 4])
        ops = [self.revnets[0],
        self.revnets[1],
        self.revnets[2],
        self.revnets[3],
        self.revnets[4],
        self.revnets[5]]
            
        if reverse:
            ops = ops[::-1]

        objective = 0.0
        for op in ops:
            x, curr_obj = op(x, reverse=reverse , training = training)
            objective += curr_obj

        x = tf.reshape(x, (-1, 64))     
        return x, objective
 


class injective(tf.keras.Model):
    def __init__(self, **kwargs):
        super(injective, self).__init__()
        """Injective subnetwork"""

        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.image_size = kwargs.get('image_size', 32)
        
        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(depth= self.depth , latent_model = False) 
        for _ in range(6)] # Bijective revnets
        
        self.inj_rev_steps = [g.revnet_step(layer_type='injective',
            latent_model = False, activation = 'linear') for _ in range(6)]
        
    def call(self, x, reverse=False , training = True):

        if reverse:
                x = tf.reshape(x, [-1, 4, 4, 4])
        ops = [
        self.squeeze,
        self.revnets[0],
        self.inj_rev_steps[0],
        self.squeeze,
        self.revnets[1],
        self.inj_rev_steps[1],
        self.squeeze,
        self.revnets[2],
        self.inj_rev_steps[2],
        self.revnets[3],
        self.inj_rev_steps[3],
        ]

        if self.image_size == 64:
            
            ops += [self.inj_rev_steps[4],
            self.revnets[4],
            self.squeeze,
            self.inj_rev_steps[5],
            self.revnets[5]
            ]
   
        if reverse:
            ops = ops[::-1]

        objective = 0.0
        for op in ops:
            x, curr_obj = op(x, reverse= reverse , training = training)
            objective += curr_obj

        if not reverse:
            x = tf.reshape(x, (-1, 4*4*4))

        return x, objective
