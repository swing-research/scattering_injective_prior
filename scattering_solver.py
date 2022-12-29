from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from utils import *
import imageio

class Operator(layers.Layer):
    """Base class of operators"""

    def __init__(self):
        super(Operator, self).__init__()
        self.opname = 'opname'

    def call(self, x):
        return x

    def T(self, x):
        return x

    def save(self, path):
        return None
 
class Inverse_scattering(Operator):
    """Builds an inverse scattering problem"""
    def __init__(self,n_inc_wave=4 , er = 3 , image_size = 64):
        super(Inverse_scattering, self).__init__()
        
        if image_size == 32:
            setup = np.load('/raid/Amir/Projects/datasets/ISP/setups/setup_32_2.0.npz')
            Gd = setup['Gd']
            Gs = setup['Gs']
            Ei = setup['Ei']
            
        elif image_size == 64:
            setup = np.load('/raid/Amir/Projects/datasets/ISP/setups/setup_64_2.0.npz')
            Gd = setup['Gd']
            Gs = setup['Gs']
            Ei = setup['Ei']
        
        Ei = Ei[:,::(np.shape(Ei)[1]//n_inc_wave)]
        Gs = Gs[::(np.shape(Gs)[0] // n_inc_wave), :]
        
        print('Ei shape:{}'.format(np.shape(Ei)))
        print('Gd shape:{}'.format(np.shape(Gd)))
        print('Gs shape:{}'.format(np.shape(Gs)))
        
        self.Gd = tf.convert_to_tensor(Gd, tf.complex64)
        self.Gs = tf.convert_to_tensor(Gs, tf.complex64)
        self.Ei = tf.convert_to_tensor(Ei, tf.complex64)
        self.n = n_inc_wave
        self.chai = er - 1
        self.opname = 'IS_%d'%self.n

    def call(self, x):
        b, h, w, c = x.get_shape().as_list()
        x = x*(self.chai/2) + self.chai/2
        x = tf.cast(x , dtype = self.Gd.dtype)
        # gt = tf.reshape(x , [-1 , np.prod(tf.shape(x)[1:])])
        gt = tf.reshape(x , [b , -1])
        I_GdGt  = tf.expand_dims(tf.eye(h*w*c , dtype = self.Gd.dtype) , axis = 0) - tf.expand_dims(self.Gd , axis = 0) * tf.expand_dims(gt , axis = 1)
        Et = tf.matmul(tf.linalg.inv(I_GdGt) , self.Ei)
        
        Gsgt = tf.expand_dims(self.Gs , axis = 0) * tf.expand_dims(gt , axis = 1)
        Es = tf.matmul(Gsgt, Et)
        return Es

    def BP(self, y):
        
        n_samples = tf.shape(y)[0]
        M = tf.shape(self.Gs)[1]
        N = int(np.sqrt(M))
        J = tf.Variable(tf.zeros([M , self.n , n_samples] , dtype = y.dtype) , trainable = False)
        Et = tf.Variable(tf.zeros([M , self.n , n_samples] , dtype = y.dtype) , trainable = False)
        
        y = tf.transpose(y , perm = [1,2,0])
        for i in range(self.n):
        # for i in range(1):
            
            A = y[:,i,:]
            B = tf.matmul(tf.matmul(self.Gs , tf.math.conj(tf.transpose(self.Gs))) , y[:,i,:])
            
            gamma = tf.reduce_sum(A * tf.math.conj(B) , axis = 0)/tf.cast(tf.reduce_sum(tf.square(tf.math.abs(B)) , axis = 0) , dtype = self.Gd.dtype)
            J[:,i,:].assign(gamma * tf.matmul(tf.math.conj(tf.transpose(self.Gs)) , y[:,i,:]))
            Et[:,i,:].assign(self.Ei[:,i:i+1] + tf.matmul(self.Gd , J[:,i,:]))
        
        A = tf.reduce_sum(J * tf.math.conj(Et) , axis = 1)
        B = tf.cast(tf.reduce_sum(tf.math.abs(Et)**2 , axis = 1) , self.Gd.dtype)
        
        Bp = tf.math.real(A/B)
        Bp = Bp/(tf.math.reduce_max(Bp, axis = 0 , keepdims = True)/2.0)
        Bp = Bp - 1
        Bp = tf.reshape(tf.transpose(Bp),  [n_samples , N , N , 1])
        
        return Bp
    
    


class InjFlow_PGD(object):
    """Builds a solver"""
    def __init__(self, flow, encoder, pz, operator, 
                 nsteps=1000, 
                 latent_dim = 192,
                 sample_shape = (64,64,3),
                 learning_rate=1e-3,
                 initial_guess = 'BP',
                 optimization_mode = 'data_space',
                 prob_folder = None):

        self.op = operator

        self.flow = flow
        self.encoder = encoder
        self.pz = pz

        self.nsteps = nsteps
        self.latent_dim = latent_dim
        self.sample_shape = sample_shape
        self.learning_rate = learning_rate
        self.initial_guess = initial_guess
        self.optimization_mode = optimization_mode
        self.prob_folder = prob_folder

    def __call__(self, measurements, gt , lam=0):

        def projection(x):
            z, rev_obj = self.encoder(x, reverse=False)
            zhat, flow_obj = self.flow(z, reverse=False)
            p = self.pz.prior.log_prob(zhat)
            # flow_obj = self.flow.log_prob(zhat)
            proj_x, fwd_obj = self.encoder(z, reverse=True)
            return proj_x, -rev_obj -flow_obj -p

        @tf.function
        def gradient_step_latent(x_guess_latent, measurements):
            with tf.GradientTape() as tape:
                x_guess, flow_obj = self.flow(x_guess_latent, reverse=True)
                x_guess, rev_obj = self.encoder(x_guess, reverse=True)
                p = self.pz.prior.log_prob(x_guess_latent)
                loss1 = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op(x_guess) - measurements), dtype=tf.float32)))
                loss2 = tf.reduce_sum(rev_obj +flow_obj -p)
                loss = loss1 + lam * loss2
                grads = tape.gradient(loss, [x_guess_latent])
                optimizer.apply_gradients(zip(grads, [x_guess_latent]))

            return x_guess, loss , loss1, loss2


        @tf.function
        def gradient_step_full(x_guess, measurements):
            with tf.GradientTape() as tape:
                proj_x_guess, likelihood = projection(x_guess)
                loss1 = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op(proj_x_guess) - measurements), dtype=tf.float32)))
                loss2 = tf.reduce_sum(likelihood)
                loss = loss1 + lam * loss2
                grads = tape.gradient(loss, [x_guess])
                optimizer.apply_gradients(zip(grads, [x_guess]))

            return proj_x_guess, loss, loss1, loss2   

        
        if self.optimization_mode == 'latent_space':

            if self.initial_guess == 'BP':
                BP_image = self.op.BP(measurements)
                BP_image, _ = self.encoder(BP_image, reverse=False)
                BP_image, _ = self.flow(BP_image, reverse=False)
                x_guess_latent = tf.Variable(BP_image, trainable=True)

            elif self.initial_guess == 'MOG':
                x_guess_latent = tf.repeat(tf.expand_dims(self.pz.mu, axis=0), repeats=[25], axis=0)
                x_guess_latent = tf.Variable(x_guess_latent, trainable=True)
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            show_per_iter = 1
            PSNR_plot = np.zeros([self.nsteps//show_per_iter])
            SSIM_plot = np.zeros([self.nsteps//show_per_iter])
            with tqdm(total=self.nsteps//show_per_iter) as pbar:

                for i in range(self.nsteps):
                    x_guess, loss, loss1 , loss2 = gradient_step_latent(x_guess_latent, measurements)
                    if i % show_per_iter == show_per_iter-1:  
                        psnr = PSNR(x_guess.numpy(), gt.numpy())
                        s = SSIM(x_guess.numpy(), gt.numpy())
                        pbar.set_description('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| SSIM: {:.2f}'.format(
                            loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))
                        pbar.update(1)

                        PSNR_plot[i//show_per_iter] = psnr
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(self.nsteps//show_per_iter)[:i//show_per_iter] , PSNR_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('PSNR')
                        plt.title('PSNR per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'PSNR.jpg'))
                        plt.close()

                        SSIM_plot[i//show_per_iter] = s
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(self.nsteps//show_per_iter)[:i//show_per_iter] , SSIM_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('SSIM')
                        plt.title('SSIM per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'SSIM.jpg'))
                        plt.close()

                        with open(os.path.join(self.prob_folder, 'results.txt'), 'a') as f:
                            f.write('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| ssim: {:.2f}'.format(
                                loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))
                            f.write('\n')

            return x_guess
        
        elif self.optimization_mode == 'data_space':
            if self.initial_guess == 'BP':
                x_guess = tf.Variable(self.op.BP(measurements), trainable=True)
            elif self.initial_guess == 'MOG':
                x_guess_latent = tf.repeat(tf.expand_dims(self.pz.mu, axis=0), repeats=[25], axis=0)
                x_guess, _ = self.flow(x_guess_latent, reverse=True)
                x_guess, _ = self.encoder(x_guess, reverse=True)
                x_guess = tf.Variable(x_guess, trainable=True)
                np.save('Initial_Guess.npy', x_guess[0])

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            show_per_iter = 1
            PSNR_plot = np.zeros([self.nsteps//show_per_iter])
            SSIM_plot = np.zeros([self.nsteps//show_per_iter])
            with tqdm(total=self.nsteps) as pbar:

                for i in range(self.nsteps):
                    proj_x_guess, loss, loss1 , loss2 = gradient_step_full(x_guess, measurements)

                    if i % show_per_iter == show_per_iter-1:  
                        psnr = PSNR(proj_x_guess.numpy(), gt.numpy())
                        s = SSIM(proj_x_guess.numpy(), gt.numpy())
                        pbar.set_description('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| ssim: {:.2f}'.format(
                            loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))

                        pbar.update(1)

                        PSNR_plot[i//show_per_iter] = psnr
        
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(self.nsteps//show_per_iter)[:i//show_per_iter] , PSNR_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('PSNR')
                        plt.title('PSNR per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'PSNR.jpg'))
                        plt.close()

                        SSIM_plot[i//show_per_iter] = s
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(self.nsteps//show_per_iter)[:i//show_per_iter] , SSIM_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('SSIM')
                        plt.title('SSIM per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'SSIM.jpg'))
                        plt.close()

                        with open(os.path.join(self.prob_folder, 'results.txt'), 'a') as f:
                            f.write('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| ssim: {:.2f}'.format(
                                loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))
                            f.write('\n')

            return projection(x_guess)[0]


def solver(
    testing_images,
    exp_path,
    operator,
    noise_snr,
    injective_model,
    bijective_model,
    latent_dim,
    pz,
    initial_guess,
    nsteps, 
    optimization_mode,
    er,
    lr_inv):
    """
    Args:
        testing_images (tf.Tensor): ground truth images (x) in the inverse problem
        exp_path (str): root directory in which to save inverse problem results
        operator (Operator): forward operator
        noise_snr (float): Noise snr
        injective_model (tf.keras.Model): the injective part of trumpet
        bijective_model (tf.keras.Model): the bijective part of trumpet
        pz (None, tf.distributions): the prior distribution
        initial_guess: the initialization of optimization: BP or MOG
        nsteps: Number of iterations
        optimization_mode: To do optimization in latent space or data space: {latent_space, data_space}
        er: Maximum epsilon_r of the medium
        lr_inv: learning rate
    """

    ngrid = 5
    _ , image_size , _ , c = tf.shape(testing_images)

    prob_folder = os.path.join(exp_path, 'er{}_init_{}_nsteps_{}_mode_{}'.format(er,initial_guess,nsteps,optimization_mode))
    if not os.path.exists(prob_folder):
        os.makedirs(prob_folder, exist_ok=True)

    x_projected = injective_model(injective_model(testing_images, reverse=False)[0], reverse=True)[0].numpy()
    
    x_projected = x_projected[:, :, :, ::-1].reshape(ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
        c)*127.5 + 127.5
    x_projected = x_projected.clip(0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(prob_folder, 'projection.png'),x_projected)


    solver = InjFlow_PGD(bijective_model, injective_model, pz, operator, 
        latent_dim=latent_dim, learning_rate = lr_inv ,
         initial_guess = initial_guess, nsteps = nsteps,
          optimization_mode = optimization_mode, prob_folder = prob_folder)

    measurements = solver.op(testing_images[:ngrid**2])

    n_snr = noise_snr
    noise_sigma = 10**(-n_snr/20.0)*tf.math.sqrt(tf.reduce_mean(tf.reduce_sum(
        tf.math.square(tf.math.abs(tf.reshape(measurements, (ngrid**2, -1)))) , -1)))

    noise_real = tf.random.normal(mean = 0,
                              stddev = noise_sigma,
                              shape = np.shape(measurements))/np.sqrt(np.prod(np.shape(measurements)[1:]))
    noise_imag = tf.random.normal(mean = 0,
                              stddev = noise_sigma,
                              shape = np.shape(measurements))/np.sqrt(np.prod(np.shape(measurements)[1:]))
    
    noise = tf.complex(noise_real, noise_imag)
    noise/= np.sqrt(2)
    measurements = measurements + noise

    gt = testing_images[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
        c)*127.5 + 127.5
    gt = gt.clip(0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(prob_folder, 'gt.png'),gt)

    bp = operator.BP(measurements).numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
        c)*127.5 + 127.5
    bp = bp.clip(0, 255).astype(np.uint8)
    imageio.imwrite(os.path.join(prob_folder, 'BP.png'),bp)

    if initial_guess == 'MOG':
        injflow_result = solver(measurements, testing_images , lam=0) 
    elif initial_guess == 'BP':
        injflow_result = solver(measurements, testing_images, lam=1e-2) 
    
    injflow_path = os.path.join(prob_folder, 'Reconstructions.png')
    injflow_result = injflow_result[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
        image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
        c)*127.5 + 127.5
    injflow_result = injflow_result.clip(0, 255).astype(np.uint8)
    imageio.imwrite(injflow_path,injflow_result)

