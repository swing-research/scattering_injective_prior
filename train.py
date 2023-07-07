import numpy as np
from time import time
import tensorflow as tf
from utils import *
from models import injective, bijective, prior
import scattering_solver
import imageio
import os
import config
import matplotlib.pyplot as plt

num_epochs = config.num_epochs
batch_size = config.batch_size
dataset = config.dataset
lr = config.lr
gpu_num = config.gpu_num
desc = config.desc
ml_threshold = config.ml_threshold
injective_depth = config.injective_depth
bijective_depth = config.bijective_depth
reload = config.reload
initial_guess = config.initial_guess
lr_inv = config.lr_inv
noise_snr = config.noise_snr
er = config.er
optimization_mode = config.optimization_mode
nsteps = config.nsteps

all_experiments = 'experiments/'
if os.path.exists(all_experiments) == False:

    os.mkdir(all_experiments)

# experiment path
exp_path = all_experiments + \
    dataset + '_' + 'injective_depth_%d' % (injective_depth,) + '_' + 'bijective_depth_%d'% (bijective_depth,) + '_' + desc

if os.path.exists(exp_path) == False:
    os.mkdir(exp_path)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


train_dataset , test_dataset = Dataset_preprocessing(dataset=dataset ,batch_size = batch_size)
print('Dataset is loaded: training and test dataset shape: {} {}'.
        format(np.shape(next(iter(train_dataset))), np.shape(next(iter(test_dataset)))))

_ , image_size , _ , c = np.shape(next(iter(train_dataset)))
f = 1
latent_dim = 4*f *4*f *4*c

optimizer_inj = tf.keras.optimizers.Adam(learning_rate=lr) # Optimizer of injective sub-network
optimizer_bij = tf.keras.optimizers.Adam(learning_rate=lr) # Optimizer of bijective sub-network

pz = prior(latent_dim = latent_dim)
inj_model = injective(revnet_depth = injective_depth ,
                    activation = 'linear',
                    f = f,
                    c = c,
                    image_size = image_size) # Injective network
bij_model = bijective(network = 'injective',revnet_depth = bijective_depth,
                                f = f,
                                c = c,
                                image_size = image_size) # Bijective network


# call generator once to set weights (Data dependent initialization for act norm layer)
dummy_x = next(iter(train_dataset))
dummy_z, _ = inj_model(dummy_x, reverse=False)
dummy_l_z , _ = bij_model(dummy_z, reverse=False)


ckpt = tf.train.Checkpoint(pz = pz , inj_model=inj_model,optimizer_inj=optimizer_inj,
    bij_model=bij_model,optimizer_bij= optimizer_bij)
manager = tf.train.CheckpointManager(
    ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

if reload:
    ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint and reload:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

if config.run_train:

    print('Training...')
    print('---> num_epochs: {}'.format(num_epochs))
    print('---> batch_size: {}'.format(batch_size))
    print('---> dataset: {}'.format(dataset))
    print('---> Learning rate: {}'.format(lr))
    print('---> experiment path: {}'.format(exp_path))

    z_inters = np.zeros([len(list(train_dataset)) * batch_size , latent_dim])
    for epoch in range(num_epochs):
        epoch_start = time()
        if epoch < ml_threshold:
            # MSE traiing of the injective network for ml-threshold epochs
            for x in train_dataset:
                train_step_mse(x, inj_model, optimizer_inj)
                
                ml_loss = 0
                p = 0
                j = 0
        
        elif epoch == ml_threshold:
            if ml_threshold == 0:
                ml_loss = 0
            counter = 0
            for x in train_dataset:
                z_inter, _ = inj_model(x, reverse= False)
                z_inters[counter*batch_size:(counter+1)*batch_size] = z_inter.numpy()
                counter = counter + 1
            
            z_inters = tf.convert_to_tensor(z_inters, tf.float32)
            z_inters_dataset = tf.data.Dataset.from_tensor_slices((z_inters))
            z_inters_dataset = z_inters_dataset.shuffle(batch_size * 3).batch(batch_size , drop_remainder = True).prefetch(5)
            
        else:
            # ML training of the bijective network after ml threshold epochs
            for x in z_inters_dataset:
                ml_loss = train_step_ml(x, bij_model, pz, optimizer_bij).numpy()
                
        if epoch == 0:
            # Show the number of trainable parametrs
            with tf.GradientTape() as tape:
                z , _ = inj_model(x, reverse= False)
                variables_inj_model = tape.watched_variables()
            
            with tf.GradientTape() as tape:
                _, _ = bij_model(z, reverse=False)
                variables_bij_model = tape.watched_variables()

            parameters_inj_model = np.sum([np.prod(v.get_shape().as_list()) for v in variables_inj_model])
            parameters_bij_model = np.sum([np.prod(v.get_shape().as_list()) for v in variables_bij_model])
            print('Number of trainable parameters of injective model: {}'.format(parameters_inj_model))
            print('Number of trainable parameters of bijective model: {}'.format(parameters_bij_model))
            print('Total number of trainable parameters: {}'.format(parameters_inj_model + parameters_bij_model))
    

        sample_number = 25 # Number of samples to show
        test_gt = next(iter(test_dataset))[:sample_number]

        # Reconstrctions
        z_test = inj_model(test_gt[:sample_number], reverse= False)[0] 
        test_recon = inj_model(z_test , reverse = True)[0].numpy()[:sample_number]
        psnr = PSNR(test_gt.numpy(), test_recon)
        
        # Sampling
        z_base = pz.prior.sample(sample_number) # sampling from base (Gaussian) 
        z_inter = bij_model(z_base , reverse = True)[0] # Intermediate samples 
        generated_samples = inj_model(z_inter , reverse = True)[0].numpy() # Randmly generated samples
        
        # Saving experiment results
        samples_folder = os.path.join(exp_path, 'Results')
        if not os.path.exists(samples_folder):
            os.mkdir(samples_folder)
        image_path_reconstructions = os.path.join(
            samples_folder, 'Reconstructions')

        image_path_generated = os.path.join(samples_folder, 'Generated samples')
        if os.path.exists(image_path_generated) == False:
            os.mkdir(image_path_generated)

        if not os.path.exists(image_path_reconstructions):
            os.mkdir(image_path_reconstructions)

        ngrid = int(np.sqrt(sample_number))
        test_recon = test_recon[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        test_recon = test_recon.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_reconstructions, '%d_recon.png' % (epoch,)),
            test_recon) # Reconstructed test images
        
        test_gt = test_gt.numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,image_size,
            image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)* 127.5 + 127.5
        test_gt = test_gt.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_reconstructions, '%d_gt.png' % (epoch,)),
            test_gt) # Ground truth test images
        
        generated_samples = generated_samples[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        generated_samples = generated_samples.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_generated, '%d_samples.png' % (epoch,)),
            generated_samples) # Generated samples

        epoch_end = time()       
        ellapsed_time = epoch_end - epoch_start
        print("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}| ML Loss: {:.3f}"
                .format(epoch, ellapsed_time, psnr, ml_loss))
            
        with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
            f.write("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}| ML Loss: {:.3f}"
                .format(epoch, ellapsed_time, psnr, ml_loss))
            f.write('\n')
        
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))



print('Solving inverse scattering problem ...:')
print('---> dataset: {}'.format(dataset))
print('---> experiment path: {}'.format(exp_path))
print('---> epsilon_r: {}'.format(er))
print('---> noise_snr: {}'.format(noise_snr))
print('---> initial guess: {}'.format(initial_guess))
print('---> learning rate of inverse problem solver:{}'.format(lr_inv))
print('---> optimization mode:{}'.format(optimization_mode))

n_test = 1 if config.experiment == 'real' else config.num_test
testing_images = next(iter(test_dataset))[:n_test]
operator = scattering_solver.Inverse_scattering(n_inc_wave = 12)

scattering_pipeline = scattering_solver.scattering(exp_path, operator, inj_model, bij_model, pz= pz)

if config.experiment == 'real':
    setup = np.load('scattering_config/fresnel_1_GtEs.npz')
    # setup = np.load('scattering_config/fresnel_2_GtEs.npz')
    testing_images = setup['gt']
    testing_images = cv2.resize(testing_images , (config.img_size,config.img_size))
    testing_images = testing_images[None,...][...,None]
    testing_images_write = (config.er-1) * ((testing_images + 1)/2) + 1

    gt_path = os.path.join(scattering_pipeline.prob_folder, f'gt_real.png')
    plt.imshow(testing_images_write[0,:,:,0], cmap = config.cmap)
    plt.colorbar()
    plt.savefig(gt_path)
    plt.close()
                

    testing_images = tf.convert_to_tensor(testing_images)
    Es = setup['Es']
    measurements = tf.convert_to_tensor(Es[None,...], tf.complex64)

else:
    measurements = scattering_pipeline.forward_solver(testing_images)


if config.run_mape:
    if config.optimization_mode == 'latent_space':
        MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images , lam=0) 
    elif config.optimization_mode == 'data_space':
        MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images, lam=1e-2) 

if config.run_posterior_sampling:
    scattering_pipeline.posterior_sampling(measurements[config.test_nb:config.test_nb+1], testing_images[config.test_nb:config.test_nb+1])
    # scattering_pipeline.laplace(measurements[config.test_nb:config.test_nb+1], testing_images[config.test_nb:config.test_nb+1])

    
    