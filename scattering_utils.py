from tqdm import tqdm
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
import config
from time import time
import tensorflow_probability as tfp



class scattering_op(object):
    """Builds an inverse scattering problem"""
    def __init__(self,n_inc_wave=4):
        
        setup = np.load(f'scattering_config/setup_{config.scattering_data}_{config.img_size}.npz')
        Gd = setup['Gd']
        Gs = setup['Gs']
        Ei = setup['Ei']

        if not config.scattering_data == 'real':
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

    def forward_op(self, x):
        # Running the forward operator

        # x is the normalized medium between -1 and 1
        b, h, w, c = x.get_shape().as_list()
        x = x*(self.chai/2) + self.chai/2 # Bring the medium to the specified contrast
        x = tf.cast(x , dtype = self.Gd.dtype)
        # gt = tf.reshape(x , [-1 , np.prod(tf.shape(x)[1:])])
        gt = tf.reshape(x , [b , -1])
        I_GdGt  = tf.expand_dims(tf.eye(h*w*c , dtype = self.Gd.dtype) , axis = 0) - tf.expand_dims(self.Gd , axis = 0) * tf.expand_dims(gt , axis = 1)
        # Et = tf.matmul(tf.linalg.inv(I_GdGt) , self.Ei)
        Et = tf.linalg.solve(I_GdGt, tf.repeat(self.Ei[None,...], b, axis = 0))
        Gsgt = tf.expand_dims(self.Gs , axis = 0) * tf.expand_dims(gt , axis = 1)
        Es = tf.matmul(Gsgt, Et)
        return Es
    

    def BP(self, y):
        # Computing back-propagation
        
        n_samples = tf.shape(y)[0]
        M = tf.shape(self.Gs)[1]
        N = int(np.sqrt(M))
        J = tf.Variable(tf.zeros([M , self.n , n_samples] , dtype = y.dtype) , trainable = False)
        Et = tf.Variable(tf.zeros([M , self.n , n_samples] , dtype = y.dtype) , trainable = False)
        
        y = tf.transpose(y , perm = [1,2,0])
        for i in range(self.n):
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
    


class scattering_solver(object):
    
    def __init__(self, exp_path, scattering_op, injective_model, bijective_model, pz):
        """
        Args:
            testing_images (tf.Tensor): ground truth images (x) in the inverse problem
            exp_path (str): root directory in which to save inverse problem results
            operator (Operator): forward operator
            injective_model (tf.keras.Model): the injective part of trumpet
            bijective_model (tf.keras.Model): the bijective part of trumpet
            pz (None, tf.distributions): the prior distribution
        """

        prob_folder = os.path.join(exp_path, 'er{}_{}_{}_{}'.format(config.er,
                                                                    config.initial_guess,
                                                                    config.solver,
                                                                    config.problem_desc))

        os.makedirs(prob_folder, exist_ok=True)

        self.op = scattering_op
        self.flow = bijective_model
        self.encoder = injective_model
        self.pz = pz
        self.prob_folder = prob_folder

    
    def forward_solver(self, testing_images):

        n_test_samples , image_size , _ , c = tf.shape(testing_images)
        ngrid = int(np.sqrt(n_test_samples))
        _ , image_size , _ , c = tf.shape(testing_images)

        x_projected = self.encoder(self.encoder(testing_images, reverse=False)[0], reverse=True)[0]
        
        x_projected = x_projected[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)
        x_projected = (config.er-1) * ((x_projected + 1)/2) + 1
        plt.imsave(os.path.join(self.prob_folder, 'projection.png'), x_projected[:,:,0], cmap = config.cmap)

        measurements = self.op.forward_op(testing_images[:ngrid**2])
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
            c)
        gt = (config.er-1) * ((gt + 1)/2) + 1
        plt.imsave(os.path.join(self.prob_folder, 'GT.png'),gt[:,:,0], cmap = config.cmap)

        bp = self.op.BP(measurements).numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)
        bp = (config.er-1) * ((bp + 1)/2) + 1
        plt.imsave(os.path.join(self.prob_folder, 'BP.png'), bp[:,:,0], cmap = config.cmap)

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

        @tf.function
        def gradient_step_latent(x_guess_latent, measurements):
            with tf.GradientTape() as tape:
                x_guess, _ = self.flow(x_guess_latent, reverse=True)
                x_guess, _ = self.encoder(x_guess, reverse=True)
                loss_data = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op.forward_op(x_guess) - measurements), dtype=tf.float32)))
                tv_loss = tf.reduce_sum(tf.image.total_variation(x_guess))
                loss = loss_data + config.tv_weight * tv_loss
                grads = tape.gradient(loss, [x_guess_latent])
                optimizer.apply_gradients(zip(grads, [x_guess_latent]))

            return x_guess, loss , loss_data, tv_loss
        
        
        @tf.function
        def gradient_step_data(x_guess, measurements):
            with tf.GradientTape() as tape:
                proj_x_guess, likelihood = projection(x_guess)
                loss_data = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op.forward_op(proj_x_guess) - measurements), dtype=tf.float32)))
                loss_likelihood = tf.reduce_sum(likelihood)
                loss = loss_data + lam * loss_likelihood
                grads = tape.gradient(loss, [x_guess])
                optimizer.apply_gradients(zip(grads, [x_guess]))

            return proj_x_guess, loss, loss_data, loss_likelihood
        

        @tf.function
        def lbfgs(x_guess_latent, measurements):

            @tf.function
            def forward(x_guess_latent):

                x_guess_latent = x_guess_latent[None, ...]

                x_guess, flow_obj = self.flow(x_guess_latent, reverse=True)
                x_guess, rev_obj = self.encoder(x_guess, reverse=True)
                p = self.pz.prior.log_prob(x_guess_latent)
                loss1 = tf.reduce_sum(tf.square(tf.cast(
                    tf.norm(self.op.forward_op(x_guess) - measurements), dtype=tf.float32)))
                loss2 = tf.reduce_sum(rev_obj +flow_obj -p)
                tv_loss = tf.reduce_sum(tf.image.total_variation(x_guess))
                loss = loss1 + lam * loss2 + config.tv_weight * tv_loss

                return loss

            @tf.function  
            def loss_and_gradient(x):
                return tfp.math.value_and_gradient(lambda x:  forward(x), x)


            result = tfp.optimizer.bfgs_minimize(loss_and_gradient,
                                                 initial_position=x_guess_latent[0],
                                                 max_iterations=10)
            x_guess_latent_optimized = result.position
            final_objective_value = result.objective_value

            x_guess, _ = self.flow(x_guess_latent_optimized, reverse=True)
            x_guess, _ = self.encoder(x_guess, reverse=True)


            return x_guess, final_objective_value
  

        
        if config.solver == 'lso':

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
                ckpt, os.path.join(self.prob_folder, 'MAP_checkpoints'), max_to_keep=3)

            if config.reload_solver:
                ckpt.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint and config.reload_solver:
                print("Solver is restored from {}".format(manager.latest_checkpoint))
            else:
                print("Solver is initialized from scratch.")

            show_per_iter = 1
            PSNR_plot = np.zeros([config.nsteps//show_per_iter])
            SSIM_plot = np.zeros([config.nsteps//show_per_iter])

            if config.optimizer == 'lbfgs':
                start = time()
                x_guess, loss = lbfgs(x_guess_latent, measurements)
                end = time()
                print(f'Ellapsed time: {end - start}')
                
                psnr = PSNR(gt.numpy(), x_guess.numpy())
                s = SSIM(gt.numpy(), x_guess.numpy())

                x_guess_write = x_guess[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                    image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                    c)
                x_guess_write = (config.er-1) * ((x_guess_write + 1)/2) + 1
                
                plt.imsave(recon_path, x_guess_write[:,:,0], cmap = config.cmap)

                with open(os.path.join(self.prob_folder, 'results.txt'), 'a') as f:
                    f.write('Loss: {:.2f} | PSNR: {:.2f}| ssim: {:.2f}'.format(
                        loss.numpy(), psnr, s))
                    f.write('\n')
        
            else: 
                with tqdm(total=config.nsteps//show_per_iter) as pbar:

                    for i in range(config.nsteps):

                        x_guess, loss, loss_data, loss_tv = gradient_step_latent(x_guess_latent, measurements)
                        if i % show_per_iter == show_per_iter-1:  
                            psnr = PSNR(gt.numpy(), x_guess.numpy())
                            s = SSIM(gt.numpy(), x_guess.numpy())
                            pbar.set_description('Loss: {:.2f}| Data: {:.2f}| TV: {:.2f} | PSNR: {:.2f}| SSIM: {:.2f}'.format(
                                loss.numpy(), loss_data.numpy(), loss_tv.numpy(), psnr, s))
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

                            np.save(os.path.join(self.prob_folder,'MAP.npy'), x_guess_latent.numpy())
                            recon_path = os.path.join(self.prob_folder, 'MAP.png')

                            x_guess_write = x_guess[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                                c)
                            x_guess_write = (config.er-1) * ((x_guess_write + 1)/2) + 1
                            plt.imsave(recon_path, x_guess_write[:,:,0], cmap = config.cmap)

                            with open(os.path.join(self.prob_folder, 'results.txt'), 'a') as f:
                                f.write('Loss: {:.2f}| Data: {:.2f}| TV: {:.2f} | PSNR: {:.2f}| SSIM: {:.2f}'.format(
                                loss.numpy(), loss_data.numpy(), loss_tv.numpy(), psnr, s))
                                f.write('\n')
                            
                            manager.save()


        
        elif config.solver == 'dso':
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
                ckpt, os.path.join(self.prob_folder, 'MAP_checkpoints'), max_to_keep=3)

            if config.reload_solver:
                ckpt.restore(manager.latest_checkpoint)

            if manager.latest_checkpoint and config.reload_solver:
                print("Solver is restored from {}".format(manager.latest_checkpoint))
            else:
                print("Solver is initialized from scratch.")            

            show_per_iter = 30
            PSNR_plot = np.zeros([config.nsteps//show_per_iter])
            SSIM_plot = np.zeros([config.nsteps//show_per_iter])
            with tqdm(total=config.nsteps//show_per_iter) as pbar:

                for i in range(config.nsteps):

                    proj_x_guess, loss, loss_data , loss_likelihood = gradient_step_data(x_guess, measurements)

                    if i % show_per_iter == show_per_iter-1:  
                        psnr = PSNR(gt.numpy(), proj_x_guess.numpy())
                        s = SSIM(gt.numpy(), proj_x_guess.numpy())
                        pbar.set_description('Loss: {:.2f}| Data: {:.2f}| NLL: {:.2f} | PSNR: {:.2f}| ssim: {:.2f}'.format(
                            loss.numpy(), loss_data.numpy(), loss_likelihood.numpy(), psnr, s))

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
                            f.write('Loss: {:.2f}| Data: {:.2f}| NLL: {:.2f} | PSNR: {:.2f}| ssim: {:.2f}'.format(
                            loss.numpy(), loss_data.numpy(), loss_likelihood.numpy(), psnr, s))
                            f.write('\n')

                        manager.save()


            x_guess = projection(x_guess)[0]
            n_test_samples , image_size , _ , c = tf.shape(gt)
            ngrid = int(np.sqrt(n_test_samples))
            recon_path = os.path.join(self.prob_folder, 'Reconstructions.png')
            x_guess_write = x_guess[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                c)
            
            x_guess_write = (config.er-1) * ((x_guess_write + 1)/2) + 1
            plt.imsave(recon_path, x_guess_write[:,:,0], cmap = config.cmap)

        return x_guess


    def posterior_sampling(self, measurements, gt):
        
        @tf.function
        def gradient_step_posterior(mape, log_sigma_q , measurements, beta):
            with tf.GradientTape() as tape:
                epsilon = tf.random.normal(mape.shape)
                z = mape + epsilon * tf.exp(log_sigma_q)
                z, _ = self.flow(z, reverse=True)
                x, _ = self.encoder(z, reverse=True)
                
                loss_data = tf.reduce_mean(tf.square(tf.cast(
                    tf.norm(self.op.forward_op(x) - measurements), dtype=tf.float32)))
                
                loss_kl = tf.reduce_mean(tf.reduce_sum(tf.square(tf.exp(log_sigma_q)) - 2*log_sigma_q, axis = 1))
                loss_tv = tf.reduce_sum(tf.image.total_variation(x))
                loss = loss_data + beta*loss_kl + config.tv_weight * loss_tv
                grads = tape.gradient(loss, [log_sigma_q])
                optimizer_posterior.apply_gradients(zip(grads, [log_sigma_q]))

            return loss , loss_data, loss_kl, loss_tv
        

        posterior_folder = os.path.join(self.prob_folder, f'Posterior_{config.beta}_{config.test_nb}')
        os.makedirs(posterior_folder, exist_ok=True)

        mape = np.load(os.path.join(self.prob_folder,'MAP.npy'))[config.test_nb:config.test_nb+1]
        mape = tf.convert_to_tensor(mape, dtype = tf.float32)
        log_sigma_q = tf.zeros(mape.shape)
        mape = tf.Variable(mape, trainable= False)
        log_sigma_q = tf.Variable(log_sigma_q, trainable=True)
        optimizer_posterior = tf.keras.optimizers.Adam(learning_rate=config.lr_posterior)

        # Checkpoints of the posterior
        ckpt = tf.train.Checkpoint(log_sigma_q = log_sigma_q, optimizer_posterior= optimizer_posterior, mape = mape)
        manager = tf.train.CheckpointManager(
            ckpt, os.path.join(posterior_folder, f'posterior_checkpoints'), max_to_keep=3)

        if config.reload_posterior:
            ckpt.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint and config.reload_posterior:
            print("Posterior is restored from {}".format(manager.latest_checkpoint))
        else:
            print("Posterior is initialized from scratch.")           

        show_per_iter = 100
        num_posterior = 25

        with tqdm(total=config.nsteps_posterior//show_per_iter) as pbar:
            for i in range(config.nsteps_posterior):
                loss, loss_data , loss_kl, loss_tv = gradient_step_posterior(mape,
                                                                             log_sigma_q,measurements,
                                                                             beta = config.beta)
                if i % show_per_iter == show_per_iter-1:  
                    manager.save()
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

                    pbar.set_description('Loss: {:.2f}| data: {:.2f}| KL: {:.2f}| TV: {:.2f}| PSNR_mmse: {:.2f}| SSIM_mmse: {:.2f} | PSNR_mape: {:.2f}| SSIM_mape: {:.2f}'.format(
                        loss.numpy(), loss_data.numpy(), loss_kl.numpy(), loss_tv.numpy(), psnr_mmse, s_mmse, psnr_map, s_map))
                    pbar.update(1)

                    with open(os.path.join(posterior_folder, 'results.txt'), 'a') as f:
                        f.write('Loss: {:.2f}| data: {:.2f}| KL: {:.2f}| TV: {:.2f}| PSNR_mmse: {:.2f}| SSIM_mmse: {:.2f} | PSNR_mape: {:.2f}| SSIM_mape: {:.2f}'.format(
                        loss.numpy(), loss_data.numpy(), loss_kl.numpy(), loss_tv.numpy(), psnr_mmse, s_mmse, psnr_map, s_map))
                        f.write('\n')
   
                    _ , image_size , _ , c = tf.shape(gt)
                    ngrid = int(np.sqrt(num_posterior))
                    posterior_path = os.path.join(posterior_folder, 'posterior_samples.png')

                    posterior_samples_write = posterior_samples[:, :, :, ::-1].numpy().reshape(ngrid, ngrid,
                        image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
                        c)
                    
                    posterior_samples_write = (config.er-1) * ((posterior_samples_write + 1)/2) + 1
                    plt.imsave(posterior_path,posterior_samples_write[:,:,0], cmap = config.cmap)

                    mmse = (config.er-1) * ((mmse + 1)/2) + 1
                    plt.imsave(os.path.join(posterior_folder, f'mmse.png'),
                                mmse[0,:,:,0],
                                cmap = config.cmap)
                    
                    plt.imsave(os.path.join(posterior_folder, 'uq.png'), uq[0,:,:,0],
                                cmap = config.cmap)
                    
                    np.savez(os.path.join(posterior_folder, 'posterior_results.npz'),
                                gt = gt.numpy()[0,:,:,0],
                                mape = mape_data.numpy()[0,:,:,0], uq = uq[0,:,:,0],
                                mmse =  mmse[0,:,:,0],
                                posterior_samples = posterior_samples_write)

                    
                  
