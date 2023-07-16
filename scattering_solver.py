from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt
from utils import *
import imageio
import config
from time import time


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
    def __init__(self,n_inc_wave=4):
        super(Inverse_scattering, self).__init__()
        
        setup = np.load(f'scattering_config/setup_{config.experiment}_{config.img_size}.npz')
        Gd = setup['Gd']
        Gs = setup['Gs']
        Ei = setup['Ei']

        if not config.experiment == 'real':
            Ei = Ei[:,::(np.shape(Ei)[1]//n_inc_wave)]
            Gs = Gs[::(np.shape(Gs)[0] // n_inc_wave), :]

        # Ei = Ei[:,:180:(np.shape(Ei)[1]//(2*n_inc_wave))]
        # Gs = Gs[:180:(np.shape(Gs)[0] // (2*n_inc_wave)), :]

        print('Ei shape:{}'.format(np.shape(Ei)))
        print('Gd shape:{}'.format(np.shape(Gd)))
        print('Gs shape:{}'.format(np.shape(Gs)))
        
        self.Gd = tf.convert_to_tensor(Gd, tf.complex64)
        self.Gs = tf.convert_to_tensor(Gs, tf.complex64)
        self.Ei = tf.convert_to_tensor(Ei, tf.complex64)
        self.n = n_inc_wave
        self.chai = config.er - 1
        self.opname = 'IS_%d'%self.n

    def call(self, x):
        # x is the normalized medium between -1 and 1
        b, h, w, c = x.get_shape().as_list()
        x = x*(self.chai/2) + self.chai/2 # Bring the medium to the specified contrast
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
    



class scattering(object):
    
    def __init__(self, exp_path, operator, injective_model, bijective_model, pz):
        """
        Args:
            testing_images (tf.Tensor): ground truth images (x) in the inverse problem
            exp_path (str): root directory in which to save inverse problem results
            operator (Operator): forward operator
            injective_model (tf.keras.Model): the injective part of trumpet
            bijective_model (tf.keras.Model): the bijective part of trumpet
            pz (None, tf.distributions): the prior distribution
        """

        prob_folder = os.path.join(exp_path, 'er{}_init_{}_mode_{}_{}'.format(config.er,
                                                                              config.initial_guess,
                                                                              config.optimization_mode,
                                                                              config.problem_desc))
        if not os.path.exists(prob_folder):
            os.makedirs(prob_folder, exist_ok=True)

        self.op = operator
        self.flow = bijective_model
        self.encoder = injective_model
        self.pz = pz
        self.prob_folder = prob_folder

    
    def forward_solver(self, testing_images):

        n_test_samples , image_size , _ , c = tf.shape(testing_images)
        ngrid = int(np.sqrt(n_test_samples))
        _ , image_size , _ , c = tf.shape(testing_images)

        x_projected = self.encoder(self.encoder(testing_images, reverse=False)[0], reverse=True)[0].numpy()
        
        x_projected = x_projected[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        x_projected = x_projected.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(self.prob_folder, 'projection.png'),x_projected[:,:,0])


        measurements = self.op(testing_images[:ngrid**2])

        n_snr = config.noise_snr
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
        imageio.imwrite(os.path.join(self.prob_folder, 'gt.png'),gt[:,:,0])

        bp = self.op.BP(measurements).numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        bp = bp.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(self.prob_folder, 'BP.png'),bp[:,:,0])

        return measurements

    def MAP_estimator(self, measurements, gt , lam=0):

        @tf.function
        def projection(x):
            # Project the data on the Trumpet manifold
            z, _ = self.encoder(x, reverse=False)
            zhat, flow_obj = self.flow(z, reverse=False)
            p = self.pz.prior.log_prob(zhat)
            proj_x, _ = self.encoder(z, reverse=True)
            return proj_x, -p -flow_obj

        # @tf.function
        def gradient_step_latent(x_guess_latent, measurements):
            with tf.GradientTape() as tape:
                x_guess, flow_obj = self.flow(x_guess_latent, reverse=True)
                x_guess, rev_obj = self.encoder(x_guess, reverse=True)
                p = self.pz.prior.log_prob(x_guess_latent)
                loss1 = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op(x_guess) - measurements), dtype=tf.float32)))
                loss2 = tf.reduce_sum(rev_obj +flow_obj -p)
                tv_loss = tf.reduce_sum(tf.image.total_variation(x_guess))
                loss = loss1 + lam * loss2 + config.tv_weight * tv_loss
                grads = tape.gradient(loss, [x_guess_latent])
                optimizer.apply_gradients(zip(grads, [x_guess_latent]))

            return x_guess, loss , loss1, loss2, tv_loss
        

        # @tf.function
        def gradient_step_latent_fast(x_guess_latent, measurements):
            b = x_guess_latent.shape[0]
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(x_guess_latent)
                t1 = time()
                x_guess, flow_obj = self.flow(x_guess_latent, reverse=True)
                x_guess, rev_obj = self.encoder(x_guess, reverse=True)
                p = self.pz.prior.log_prob(x_guess_latent)
                t2 = time()
                print(t2-t1)
                data_shape = x_guess.shape
                x_guess_reshaped = tf.reshape(x_guess, [b, -1])
                x_guess = tf.reshape(x_guess_reshaped, data_shape)
                
                y = self.op(x_guess)
                t3 = time()
                print(t3 - t2)
                l_data = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(y - measurements), dtype=tf.float32)))
                l_likelihood = tf.reduce_sum(rev_obj +flow_obj -p)
                l_tv = tf.reduce_sum(tf.image.total_variation(x_guess))
                loss = l_data + lam * l_likelihood + config.tv_weight * l_tv
                
            s = time()
            j_f = tape.batch_jacobian(x_guess_reshaped, x_guess_latent) 
            s1 = time()
            print(j_f.shape, s1 - s)
            j_A = tape.batch_jacobian(y, x_guess_reshaped)
            s2 = time()
            print(j_A.shape, s2 - s1)
            j_l = tape.gradient(l_data, y)
            s3 = time()
            print(j_l.shape, s3 - s2)
            g_data = j_l
            g_likelihood = tape.gradient(l_likelihood, x_guess_latent)
            g_tv = tape.gradient(loss, l_tv, x_guess_latent)
            grads = g_data + lam * g_likelihood + config.tv_weight * g_tv
            optimizer.apply_gradients(zip(grads, [x_guess_latent]))

            return x_guess, loss , loss1, loss2, tv_loss



        @tf.function
        def gradient_step_data(x_guess, measurements):
            with tf.GradientTape() as tape:
                proj_x_guess, likelihood = projection(x_guess)
                loss1 = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op(proj_x_guess) - measurements), dtype=tf.float32)))
                loss2 = tf.reduce_sum(likelihood)
                loss = loss1 + lam * loss2
                grads = tape.gradient(loss, [x_guess])
                optimizer.apply_gradients(zip(grads, [x_guess]))

            return proj_x_guess, loss, loss1, loss2   

        
        if config.optimization_mode == 'lso':

            n_test_samples , image_size , _ , c = tf.shape(gt)
            ngrid = int(np.sqrt(n_test_samples))
            if config.initial_guess == 'BP':
                BP_image = self.op.BP(measurements)
                BP_image, _ = self.encoder(BP_image, reverse=False)
                BP_image, _ = self.flow(BP_image, reverse=False)
                x_guess_latent = tf.Variable(BP_image, trainable=True)

            elif config.initial_guess == 'MOG':
                x_guess_latent = tf.repeat(tf.expand_dims(self.pz.mu, axis=0), repeats=[gt.shape[0]], axis=0)
                x_guess_latent = tf.Variable(x_guess_latent, trainable=True)

            optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr_inv)
            
            # Checkpoints of the solver
            ckpt = tf.train.Checkpoint(x_guess_latent = x_guess_latent, optimizer= optimizer)
            manager = tf.train.CheckpointManager(
                ckpt, os.path.join(self.prob_folder, 'solver_checkpoints'), max_to_keep=3)

            if config.reload_solver:
                ckpt.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint and config.reload_solver:
                print("Solver is restored from {}".format(manager.latest_checkpoint))
            else:
                print("Solver is initialized from scratch.")

            show_per_iter = 450
            PSNR_plot = np.zeros([config.nsteps//show_per_iter])
            SSIM_plot = np.zeros([config.nsteps//show_per_iter])
            with tqdm(total=config.nsteps//show_per_iter) as pbar:

                start = time()
                for i in range(config.nsteps):
                    x_guess, loss, loss1 , loss2, tv_loss = gradient_step_latent(x_guess_latent, measurements)
                    
                    if i % show_per_iter == show_per_iter-1:  
                        psnr = PSNR(gt.numpy(), x_guess.numpy())
                        s = SSIM(gt.numpy(), x_guess.numpy())
                        pbar.set_description('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| TV: {:.2f} | PSNR: {:.2f}| SSIM: {:.2f}'.format(
                            loss.numpy(), loss2.numpy(), loss1.numpy(), tv_loss.numpy(), psnr, s))
                        pbar.update(1)

                        PSNR_plot[i//show_per_iter] = psnr
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(config.nsteps//show_per_iter)[:i//show_per_iter] , PSNR_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('PSNR')
                        plt.title('PSNR per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'PSNR.jpg'))
                        plt.close()

                        SSIM_plot[i//show_per_iter] = s
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(config.nsteps//show_per_iter)[:i//show_per_iter] , SSIM_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('SSIM')
                        plt.title('SSIM per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'SSIM.jpg'))
                        plt.close()


                        np.save(os.path.join(self.prob_folder,'MAPE.npy'), x_guess_latent.numpy())
                        recon_path = os.path.join(self.prob_folder, 'MAPE.png')

                        if config.cmap == 'gray':
                            x_guess_write = x_guess[0, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                                c)*127.5 + 127.5
                            x_guess_write = x_guess_write.clip(0, 255).astype(np.uint8)
                            imageio.imwrite(recon_path,x_guess_write[:,:,0])
                        
                        else:
                            x_guess_write = x_guess[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                                c)
                            x_guess_write = (config.er-1) * ((x_guess_write + 1)/2) + 1
                            
                            plt.imshow(x_guess_write[:,:,0], cmap = config.cmap)
                            plt.colorbar()
                            plt.savefig(recon_path)
                            plt.close()

                        with open(os.path.join(self.prob_folder, 'results.txt'), 'a') as f:
                            f.write('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| ssim: {:.2f}'.format(
                                loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))
                            f.write('\n')
                        
                        manager.save()

                end = time()
                print(f'Ellapsed time: {end - start}')


        
        elif config.optimization_mode == 'dso':
            if config.initial_guess == 'BP':
                x_guess = tf.Variable(self.op.BP(measurements), trainable=True)
            elif config.initial_guess == 'MOG':
                x_guess_latent = tf.repeat(tf.expand_dims(self.pz.mu, axis=0), repeats=[gt.shape[0]], axis=0)
                x_guess, _ = self.flow(x_guess_latent, reverse=True)
                x_guess, _ = self.encoder(x_guess, reverse=True)
                x_guess = tf.Variable(x_guess, trainable=True)

            optimizer = tf.keras.optimizers.Adam(learning_rate=config.lr_inv)
            
            # Checkpoints of the solver
            ckpt = tf.train.Checkpoint(x_guess = x_guess, optimizer= optimizer)
            manager = tf.train.CheckpointManager(
                ckpt, os.path.join(self.prob_folder, 'solver_checkpoints'), max_to_keep=3)

            if config.reload_solver:
                ckpt.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint and config.reload_solver:
                print("Solver is restored from {}".format(manager.latest_checkpoint))
            else:
                print("Solver is initialized from scratch.")            

            show_per_iter = 450
            PSNR_plot = np.zeros([config.nsteps//show_per_iter])
            SSIM_plot = np.zeros([config.nsteps//show_per_iter])
            with tqdm(total=config.nsteps//show_per_iter) as pbar:

                start = time()
                for i in range(config.nsteps):
                    proj_x_guess, loss, loss1 , loss2 = gradient_step_data(x_guess, measurements)

                    if i % show_per_iter == show_per_iter-1:  
                        psnr = PSNR(gt.numpy(), proj_x_guess.numpy())
                        s = SSIM(gt.numpy(), proj_x_guess.numpy())
                        pbar.set_description('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| ssim: {:.2f}'.format(
                            loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))

                        pbar.update(1)

                        PSNR_plot[i//show_per_iter] = psnr
        
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(config.nsteps//show_per_iter)[:i//show_per_iter] , PSNR_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('PSNR')
                        plt.title('PSNR per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'PSNR.jpg'))
                        plt.close()

                        SSIM_plot[i//show_per_iter] = s
                        plt.figure(figsize=(10,6), tight_layout=True)
                        plt.plot(np.arange(config.nsteps//show_per_iter)[:i//show_per_iter] , SSIM_plot[:i//show_per_iter], 'o-', linewidth=2)
                        plt.xlabel('iteration')
                        plt.ylabel('SSIM')
                        plt.title('SSIM per {} iteration'.format(show_per_iter))
                        plt.savefig(os.path.join(self.prob_folder, 'SSIM.jpg'))
                        plt.close()

                        with open(os.path.join(self.prob_folder, 'results.txt'), 'a') as f:
                            f.write('Loss: {:.2f}| NLL: {:.2f}| Data: {:.2f}| PSNR: {:.2f}| ssim: {:.2f}'.format(
                                loss.numpy(), loss2.numpy(), loss1.numpy(), psnr, s))
                            f.write('\n')

                        manager.save()
                
                end = time()
                print(f'Ellapsed time: {end - start}')

            x_guess = projection(x_guess)[0]
            n_test_samples , image_size , _ , c = tf.shape(gt)
            ngrid = int(np.sqrt(n_test_samples))
            recon_path = os.path.join(self.prob_folder, 'Reconstructions.png')
            x_guess_write = x_guess[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                c)*127.5 + 127.5
            x_guess_write = x_guess_write.clip(0, 255).astype(np.uint8)
            imageio.imwrite(recon_path,x_guess_write[:,:,0])

        return x_guess

    def posterior_sampling(self, measurements, gt):
        
        @tf.function
        def gradient_step_VI(mape, log_sigma_q , measurements, beta):
            with tf.GradientTape() as tape:
                epsilon = tf.random.normal(mape.shape)
                z = mape + epsilon * tf.exp(log_sigma_q)
                z, _ = self.flow(z, reverse=True)
                x, _ = self.encoder(z, reverse=True)
                
                loss1 = tf.reduce_mean(tf.square(tf.cast(
                    tf.norm(self.op(x) - measurements), dtype=tf.float32)))
                
                loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.exp(log_sigma_q)) - 2*log_sigma_q, axis = 1))
                tv_loss = tf.reduce_sum(tf.image.total_variation(x))
                loss = loss1 + beta*loss2 + config.tv_weight * tv_loss
                grads = tape.gradient(loss, [log_sigma_q])
                optimizer_VI.apply_gradients(zip(grads, [log_sigma_q]))

            return x, loss , loss1, loss2, tv_loss
        
        mape = np.load(os.path.join(self.prob_folder,'MAPE.npy'))[config.test_nb:config.test_nb+1]
        mape = tf.convert_to_tensor(mape, dtype = tf.float32)
        log_sigma_q = tf.zeros(mape.shape)
        mape = tf.Variable(mape, trainable=config.mean_optimized)
        log_sigma_q = tf.Variable(log_sigma_q, trainable=True)
        optimizer_VI = tf.keras.optimizers.Adam(learning_rate=config.lr_VI)

        # Checkpoints of the VI
        ckpt = tf.train.Checkpoint(log_sigma_q = log_sigma_q, optimizer_VI= optimizer_VI, mape = mape)
        manager = tf.train.CheckpointManager(
            ckpt, os.path.join(self.prob_folder, f'VI_checkpoints_{config.beta}_{config.test_nb}_{config.mean_optimized}'), max_to_keep=3)

        if config.reload_VI:
            ckpt.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint and config.reload_VI:
            print("VI is restored from {}".format(manager.latest_checkpoint))
        else:
            print("VI is initialized from scratch.")           

        show_per_iter = 100
        num_posterior = 25

        with tqdm(total=config.nsteps_VI//show_per_iter) as pbar:
            for i in range(config.nsteps_VI):
                x, loss, loss1 , loss2, tv_loss = gradient_step_VI(mape, log_sigma_q,
                                                                measurements, beta = config.beta)
                if i % show_per_iter == show_per_iter-1:  
                    epsilon = tf.random.normal([num_posterior,mape.shape[1]], seed = 0)
                    z = mape + epsilon * tf.exp(log_sigma_q)
                    z, _ = self.flow(z, reverse=True)
                    posterior_samples, _ = self.encoder(z, reverse=True)
                    mmse =  posterior_samples.numpy().mean(axis = 0, keepdims = True)
                    uq = posterior_samples.numpy().std(axis = 0, keepdims = True)
                    psnr_mmse = PSNR(gt.numpy(),mmse)
                    s_mmse = SSIM(gt.numpy(), mmse)

                    mape_data, _ =  self.flow(mape, reverse=True)
                    mape_data, _ = self.encoder(mape_data, reverse=True)
                    psnr_map = PSNR(gt.numpy(),mape_data.numpy())
                    s_map = SSIM(gt.numpy(), mape_data.numpy())

                    pbar.set_description('Loss: {:.2f}| data: {:.2f}| kl: {:.2f}| TV: {:.2f}| PSNR_mmse: {:.2f}| SSIM_mmse: {:.2f} | PSNR_mape: {:.2f}| SSIM_mape: {:.2f}'.format(
                        loss.numpy(), loss1.numpy(), loss2.numpy(), tv_loss.numpy(), psnr_mmse, s_mmse, psnr_map, s_map))
                    pbar.update(1)

                    with open(os.path.join(self.prob_folder, f'results_{config.beta}_{config.test_nb}_{config.mean_optimized}.txt'), 'a') as f:
                        f.write('Loss: {:.2f}| data: {:.2f}| kl: {:.2f}| TV: {:.2f}| PSNR_mmse: {:.2f}| SSIM_mmse: {:.2f} | PSNR_mape: {:.2f}| SSIM_mape: {:.2f}'.format(
                        loss.numpy(), loss1.numpy(), loss2.numpy(), tv_loss.numpy(), psnr_mmse, s_mmse, psnr_map, s_map))
                        f.write('\n')
   
                    _ , image_size , _ , c = tf.shape(gt)
                    ngrid = int(np.sqrt(num_posterior))
                    posterior_path = os.path.join(self.prob_folder, f'posterior_samples_{config.beta}_{config.test_nb}_{config.mean_optimized}.png')

                    if config.experiment == 'gray':
                        posterior_samples_write = posterior_samples[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                            c)*127.5 + 127.5
                        posterior_samples_write = posterior_samples_write.clip(0, 255).astype(np.uint8)
                        imageio.imwrite(posterior_path,posterior_samples_write[:,:,0])

                        mmse = mmse * 127.5 + 127.5
                        mmse = mmse.clip(0, 255).astype(np.uint8)
                        imageio.imwrite(os.path.join(self.prob_folder, f'mmse_{config.beta}_{config.test_nb}_{config.mean_optimized}.png'),mmse[0,:,:,0])

                    
                    else:
                        posterior_samples_write = posterior_samples[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                            c)
                        
                        posterior_samples_write = (config.er-1) * ((posterior_samples_write + 1)/2) + 1
                
                        plt.imshow(posterior_samples_write[:,:,0], cmap = config.cmap)
                        plt.colorbar()
                        plt.savefig(posterior_path)
                        plt.close()

                        plt.imshow(mmse[0,:,:,0], cmap = config.cmap)
                        plt.colorbar()
                        plt.savefig(os.path.join(self.prob_folder, f'mmse_{config.beta}_{config.test_nb}_{config.mean_optimized}.png'))
                        plt.close()
                        plt.imsave(os.path.join(self.prob_folder, f'uq_{config.beta}_{config.test_nb}_{config.mean_optimized}.png'), uq[0,:,:,0], cmap = 'seismic')
                        np.savez(os.path.join(self.prob_folder, f'{config.beta}_{config.test_nb}_{config.mean_optimized}.npz'),
                                 gt = gt.numpy()[0,:,:,0], mape = mape_data.numpy()[0,:,:,0], uq = uq[0,:,:,0], mmse =  mmse[0,:,:,0], posterior_samples = posterior_samples_write)
                    manager.save()

    
    def laplace(self, measurements, gt):
        mape = np.load(os.path.join(self.prob_folder,'MAPE.npy'))[config.test_nb]
        mape = tf.convert_to_tensor(mape, dtype = tf.float32)
        print(mape.shape)

        with tf.GradientTape() as g:
            g.watch(mape)
            with tf.GradientTape() as gg:
                gg.watch(mape)
                z = mape[None,...]
                z_inter, _ = self.flow(z, reverse=True)
                # x, _ = self.encoder(z_inter, reverse=True)
                x= z_inter
                l = tf.square(tf.cast(tf.norm(x), dtype=tf.float32))
                # l = tf.square(tf.cast(tf.norm(self.op(x) - measurements), dtype=tf.float32))
            
            print(l.shape)
            grads = gg.gradient(l , mape)
            print(grads.shape)
        hessian = g.jacobian(grads, mape)
        print(hessian.shape)
    

        

