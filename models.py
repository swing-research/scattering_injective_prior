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
        """ Bijective architecture"""

        self.network = kwargs.get('network', 'injective') # revnet depth
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.f = kwargs.get('f', 1)
        self.c = kwargs.get('c', 3)
        self.image_size = kwargs.get('image_size', 32)
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth

        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(coupling_type='affine', depth = self.depth , latent_model = True) 
        for _ in range(6)]

    def call(self, x, reverse=False , training = True):

        x = tf.reshape(x, [-1,4*self.f, 4*self.f, 4*self.c])
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

        x = tf.reshape(x, (-1, 4*self.f *4*self.f *4*self.c))     
        return x, objective
 



class injective(tf.keras.Model):
    def __init__(self, **kwargs):
        super(injective, self).__init__()
        """Injective architecture
        . upsqueeze
        --> revnet
        |-> inj_rev_step
        
        + 4x4x12 --> 4x4x12 |-> 4x4x24 . 8x8x6
         --> 8x8x6 |-> 8x8x12 |-> 8x8x24 --> 8x8x24
        |-> 8x8x48 --> 8x8x48 . 16x16x12 |-> 16x16x24
        --> 16x16x24 . 32x32x6 |-> 32x32x12 --> 32x32x12
        . 64x64x3
        
        summary for ellipses: 
        6 bijective revnets
        6 injective revnet_steps
        4 upsqueeze
        """
        self.depth = kwargs.get('revnet_depth', 3) # revnet depth
        self.activation = kwargs.get('activation', 'linear') # activation of invertible 1x1 convolutional layer
        self.f = kwargs.get('f', 1)
        self.c = kwargs.get('c', 3)
        self.image_size = kwargs.get('image_size', 32)
        
        self.squeeze = g.upsqueeze(factor=2)
        self.revnets = [g.revnet(depth= self.depth , latent_model = False) 
        for _ in range(6)] # Bijective revnets
        
        self.inj_rev_steps = [g.revnet_step(layer_type='injective',
            latent_model = False, activation = self.activation) for _ in range(6)]
        
    def call(self, x, reverse=False , training = True):

        if reverse:
                x = tf.reshape(x, [-1,4*self.f, 4*self.f, 4*self.c])
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
            x = tf.reshape(x, (-1, 4*self.f *4*self.f *4*self.c))

        return x, objective
