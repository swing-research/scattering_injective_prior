import tensorflow as tf
from tensorflow.keras import layers
import scipy
import numpy as np
from Unet_util import Unet


class upsqueeze(layers.Layer):
    def __init__(self, factor=2):
        super(upsqueeze, self).__init__()
        self.f = factor

    def call(self, x, reverse=False):
        f = self.f

        # upsampling via squeeze
        b, N1, N2, nch = x.get_shape().as_list()
        if not reverse:
            x = tf.reshape(
                tf.transpose(
                    tf.reshape(x, shape=[b, N1//f, f, N2//f, f, nch]),
                    [0, 1, 3, 2, 4, 5]),
                [b, N1//f, N2//f, nch*f*f])
        else:
            x = tf.reshape(tf.transpose(
                tf.reshape(x, shape=[b, N1, N2, f, f, nch//f**2]),
                [0, 1, 3, 2, 4, 5]), [b, N1*f, N2*f, nch//f**2])


        return x, 0.0



class actnorm(layers.Layer):
    """Activation normalization layers that 
    initialized via data"""

    def __init__(self, **kwargs):
        super(actnorm, self).__init__()
        
        # assign checks for first call
        self.assigned = False

    def build(self, input_shape):
        if len(input_shape) == 2:
            self.b = self.add_weight(name='bias',
                                     shape=(1, input_shape[1]),
                                     trainable= True)
            self.scale = self.add_weight(name='scale',
                                         shape=(1, input_shape[1]),
                                         trainable= True)
        else:
            self.b = self.add_weight(name='bias',
                                     shape=(1, 1, 1, input_shape[3]),
                                     trainable= True)
            self.scale = self.add_weight(name='scale',
                                         shape=(1, 1, 1, input_shape[3]),
                                         trainable= True)

    def call(self, x, reverse=False):
        if len(x.shape) == 2:
            red_axes = [0]
            dim = x.get_shape().as_list()[-1]
        else:
            red_axes = [0, 1, 2]
            _, height, width, channels = x.get_shape().as_list()
            dim = height*width
        
        if not self.assigned:
            """https://github.com/tensorflow/tensor2tensor/blob/21dba2c1bdcc7ab582a2bfd8c0885c217963bb4f/tensor2tensor/models/research/glow_ops.py#L317"""
            self.b.assign(-tf.reduce_mean(x, red_axes, keepdims=True))

            x_var = tf.reduce_mean((x+self.b)**2, red_axes, keepdims=True)
            init_value = tf.math.log(1.0/(tf.math.sqrt(x_var) + 1e-6))


            self.scale.assign(init_value)
            self.assigned = True


        if not reverse:
            x += self.b
            x *= tf.math.exp(self.scale)

        else:
            x *= tf.math.exp(-self.scale)
            x -= self.b
        
        log_s = self.scale
        dlogdet = tf.reduce_sum(log_s)* \
            tf.cast(dim, log_s.dtype)
        if reverse:
            dlogdet *= -1

        return x, dlogdet


class invertible_1x1_conv(layers.Layer):
    """Invertible 1x1 convolutional layers"""

    def __init__(self, **kwargs):
        super(invertible_1x1_conv, self).__init__()
        self.type = kwargs.get('op_type', 'bijective')
        self.gamma = kwargs.get('gamma', 0.0)
        self.activation = kwargs.get('activation', 'linear')
        
    def build(self, input_shape):
        _, height, width, channels = input_shape
        
        if self.type=='bijective':
            random_matrix = np.random.randn(channels, channels).astype("float32")
            np_w = scipy.linalg.qr(random_matrix)[0].astype("float32")
            self.activation = 'linear'
            self.LU = True
            
            if self.LU:
                np_p, np_l, np_u = scipy.linalg.lu(np_w)
                np_s = np.diag(np_u)
                np_sign_s = np.sign(np_s)
                np_log_s = np.log(np.abs(np_s))
                np_u = np.triu(np_u, k=1)
        
                self.p = tf.Variable(np_p, name='P', trainable=False)
                self.l = tf.Variable(np_l, name='L', trainable=True)
                self.sign_s = tf.Variable(np_sign_s, name='sign_S',
                                      trainable=False)
                self.log_s = tf.Variable(np_log_s, name='log_S',
                                          trainable=True)
                self.u = tf.Variable(np_u, name='U',
                                      trainable=True)
                
            else:
                
                self.w = tf.Variable(np_w, name='W', trainable=True)
                
            
        else:
            self.LU = False
            if self.activation == 'linear':
                random_matrix_1 = np.random.randn(channels//2, channels//2).astype("float32")
                random_matrix_2 = np.random.randn(channels//2, channels//2).astype("float32")
                np_w_1 = scipy.linalg.qr(random_matrix_1)[0].astype("float32")
                np_w_2 = scipy.linalg.qr(random_matrix_2)[0].astype("float32")
                np_w = np.concatenate([np_w_1, np_w_2], axis=0)/(np.sqrt(2.0))
                
            elif self.activation == 'relu':
                random_matrix_1 = np.random.randn(channels//2, channels//2).astype("float32")
                np_w = scipy.linalg.qr(random_matrix_1)[0].astype("float32")
                
            self.w = tf.Variable(np_w, name='W', trainable=True)
                
    def call(self, x, reverse=False):
        # If height or width cannot be statically determined then they end up as
        # tf.int32 tensors, which cannot be directly multiplied with a floating
        # point tensor without a cast.
        _, height, width, channels = x.get_shape().as_list()
        
        if self.type=='bijective':
            
            if self.LU:
                l_mask = tf.convert_to_tensor(np.tril(np.ones([channels, channels], dtype=np.float32), -1),
                                              dtype=tf.float32)
                
                l = self.l * l_mask + tf.eye(channels, channels)
                u = self.u * tf.transpose(l_mask) + \
                    tf.linalg.diag(self.sign_s * tf.math.exp(self.log_s))
                self.w = tf.matmul(self.p, tf.matmul(l, u))
                objective = tf.reduce_sum(self.log_s) * \
                tf.cast(height * width, self.log_s.dtype)
            
            else:
                s = tf.linalg.svd(self.w, 
                    full_matrices=False, compute_uv=False)
                self.log_s = tf.math.log(s + self.gamma**2/(s + 1e-8))
                objective = tf.reduce_sum(self.log_s) * \
                tf.cast(height * width, self.log_s.dtype)
                
        else:
            # s = tf.linalg.svd(self.w, 
            #     full_matrices=False, compute_uv=False)
            # self.log_s = tf.math.log(s + self.gamma**2/(s + 1e-8))
            objective = 0.0

        if not reverse:
            
            if self.activation == 'relu':
                x = x[:,:,:,:channels//2] - x[:,:,:,channels//2:]
            w = tf.reshape(self.w , [1, 1] + self.w.get_shape().as_list())
            x = tf.nn.conv2d(x, w, [1, 1, 1, 1], "SAME", data_format="NHWC")
        
        
        else:
            if self.activation=='relu':
                prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                    self.gamma**2*tf.eye(tf.shape(self.w)[1])
                
                w_inv = tf.matmul(tf.linalg.inv(prefactor), self.w, transpose_b=True)
                conv_filter = tf.concat([w_inv, -w_inv], axis=1)
                conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
                x = tf.nn.conv2d(x, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")
                x = tf.nn.relu(x)
            else:
                if self.LU == True:
                    perm = tf.argmax(self.p , axis = 0)
                    lower_upper = l + u - tf.eye(channels)
                    w_inv = tf.linalg.lu_matrix_inverse(lower_upper=lower_upper , perm = perm)
                    
                else:
                    prefactor = tf.matmul(self.w, self.w, transpose_a=True) + \
                        self.gamma**2*tf.eye(tf.shape(self.w)[1])
                    w_inv = tf.matmul(  tf.linalg.inv(prefactor) , self.w, transpose_b=True)
                    
                conv_filter = w_inv
                conv_filter = tf.reshape(conv_filter, [1, 1] + conv_filter.get_shape().as_list())
                x = tf.nn.conv2d(x, conv_filter, [1, 1, 1, 1], "SAME", data_format="NHWC")
            
            objective *= -1
        return x, objective





class conv_stack(layers.Layer):
    def __init__(self, mid_channels,
                 output_channels):
        super(conv_stack, self).__init__()

        self.conv1 = layers.Conv2D(
            mid_channels, 3, 1, padding='same',
            activation='relu', use_bias=False)
        self.conv2 = layers.Conv2D(
            mid_channels, 1, 1, padding='same',
            activation='relu', use_bias=False)
        self.conv3 = layers.Conv2D(
            output_channels, 1, 1, padding='same', activation='sigmoid', use_bias=False,kernel_initializer='zeros')

    def call(self, x):
        return self.conv3(self.conv2(self.conv1(x)))




class affine_coupling(layers.Layer):
    def __init__(self):
        super(affine_coupling, self).__init__()
        

    def build(self, input_shape):
        out_channels = input_shape[-1]
        # self.conv_stack = conv_stack(128,out_channels) # regular convolutions
        self.conv_stack = Unet(out_channels) # Unet conv stack

    def call(self, x, reverse=False):
        # out_ch = x.get_shape().as_list()[-1]
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        
        # alpha = 0.01
        z1 = x1
        log_scale_and_shift = self.conv_stack(z1)
        shift = log_scale_and_shift[:, :, :, 0::2]
        scale =  tf.math.exp(log_scale_and_shift[:, :, :, 1::2])
        # scale = alpha + (1-alpha)*scale # to be more stable in reverse
        if not reverse:
            z2 = (x2 + shift) *scale
            # z2 = x2 + shift
        else:
            z2 = (x2/scale) - shift
            # z2 = x2/ - shift
            

        objective = tf.reduce_sum(log_scale_and_shift[:, :, :, 1::2], axis=[1, 2, 3])
        # objective = 0.0

        if reverse:
            objective *= -1
        return tf.concat([z1, z2], axis=3), objective


class revnet_step(layers.Layer):
    """One layer of this is:
    [1] Actnorm -- data normalization
    [2] 1x1 conv -- permutation
    [3] coupling layer -- Jacobian
    """
    def __init__(self, **kwargs):
        super(revnet_step, self).__init__()
        self.layer_type = kwargs.get('layer_type', 'bijective')
        self.mid_ch = kwargs.get('mid_channels', 128)
        self.latent_model = kwargs.get('latent_model', False)
        self.activation = kwargs.get('activation', 'linear')
        self.norm = actnorm()
        
        gamma = 1e-3 if self.latent_model else 1e-3
        self.conv = invertible_1x1_conv(
            op_type=self.layer_type , activation = self.activation , gamma = gamma)
        self.coupling = affine_coupling()

    def call(self, x, reverse=False):
        obj = 0.0
        ops = [ self.norm, self.conv, self.coupling]
        
        if reverse:
            ops = ops[::-1]

        for op in ops:  
            x, curr_obj = op(x, reverse=reverse)
            obj += curr_obj

        return x, obj


class revnet(layers.Layer):
    """Composition of revnet steps"""
    def __init__(self, **kwargs):
        super(revnet, self).__init__()
        self.depth = kwargs.get('depth', 3)
        self.latent_model = kwargs.get('latent_model', False)
        self.steps = [revnet_step(layer_type = 'bijective',
                                  latent_model = self.latent_model ,
                                  activation = 'linear')
                      for _ in range(self.depth)]
        
    def call(self, x, reverse=False):
        objective = 0.0
        if reverse:
            steps = self.steps[::-1]
        else:
            steps = self.steps

        for i in range(self.depth):
            step = steps[i]
            x, curr_obj = step(x,reverse=reverse)
            objective += curr_obj

        return x, objective
    