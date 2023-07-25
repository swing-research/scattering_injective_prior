import numpy as np
from time import time
import tensorflow as tf
from utils import *
from models import injective, bijective, prior
import scattering_utils
import imageio
import os
import config
import matplotlib.pyplot as plt

all_experiments = 'experiments/'
os.makedirs(all_experiments, exist_ok=True)

# experiment path
exp_path = all_experiments + \
    f'{config.dataset}_{config.inj_depth}_{config.bij_depth}_{config.desc}'

os.makedirs(exp_path, exist_ok=True)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[config.gpu_num], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[config.gpu_num], True)
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

train_dataset , test_dataset = Dataset_preprocessing(dataset=config.dataset ,batch_size = config.batch_size)
print('Dataset is loaded: training and test dataset shape: {} {}'.
        format(np.shape(next(iter(train_dataset))), np.shape(next(iter(test_dataset)))))

_ , image_size , _ , c = np.shape(next(iter(train_dataset)))
latent_dim = 64

optimizer_inj = tf.keras.optimizers.Adam(learning_rate=config.lr) # Optimizer of injective sub-network
optimizer_bij = tf.keras.optimizers.Adam(learning_rate=config.lr) # Optimizer of bijective sub-network

pz = prior(latent_dim = latent_dim)
inj_model = injective(revnet_depth = config.inj_depth,
                      image_size = image_size) # Injective network

bij_model = bijective(revnet_depth = config.bij_depth) # Bijective network

num_params_inj_model = np.sum([np.prod(v.get_shape()) for v in inj_model.trainable_weights])
num_params_bij_model = np.sum([np.prod(v.get_shape()) for v in bij_model.trainable_weights])
print('Number of trainable parameters of injective subnetwork: {}'.format(num_params_inj_model))
print('Number of trainable parameters of bijective subnetwork: {}'.format(num_params_bij_model))

# call generator once to set weights (Data dependent initialization for act norm layer)
dummy_x = next(iter(train_dataset))
dummy_z, _ = inj_model(dummy_x, reverse=False)
dummy_l_z , _ = bij_model(dummy_z, reverse=False)

ckpt = tf.train.Checkpoint(pz = pz , inj_model= inj_model, optimizer_inj=optimizer_inj,
    bij_model=bij_model, optimizer_bij= optimizer_bij)
manager = tf.train.CheckpointManager(
    ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

if config.reload:
    ckpt.restore(manager.latest_checkpoint)

if manager.latest_checkpoint and config.reload:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

samples_folder = os.path.join(exp_path, 'Results')
os.makedirs(samples_folder, exist_ok=True)

if config.train_injective:

    print('Training of the Injective Subnetwork')
    print('---> Dataset: {}'.format(config.dataset))
    print('---> Experiment path: {}'.format(exp_path))
    print('---> Injective depth: {}'.format(config.inj_depth))
    print('---> Num epochs: {}'.format(config.n_epochs_inj))
    print('---> Learning rate: {}'.format(config.lr))

    ngrid = int(np.sqrt(config.n_test))
    image_path_reconstructions = os.path.join(
        samples_folder, 'Reconstructions')

    os.makedirs(image_path_reconstructions, exist_ok=True)

    for epoch in range(config.n_epochs_inj):
        epoch_start = time()  
        for x in train_dataset:
            train_step_mse(x, inj_model, optimizer_inj)
        
        # Reconstrctions
        test_gt = next(iter(test_dataset))[:config.n_test]
        z_test = inj_model(test_gt[:config.n_test], reverse= False)[0] 
        test_recon = inj_model(z_test , reverse = True)[0].numpy()[:config.n_test]
        psnr = PSNR(test_gt.numpy(), test_recon)

        test_recon = test_recon[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        test_recon = test_recon.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_reconstructions, '%d_recon.png' % (epoch,)),
            test_recon[:,:,0]) # Reconstructed test images
        
        test_gt = test_gt.numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,image_size,
            image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)* 127.5 + 127.5
        test_gt = test_gt.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_reconstructions, '%d_gt.png' % (epoch,)),
            test_gt[:,:,0]) # Ground truth test images
        
        epoch_end = time()       
        ellapsed_time = epoch_end - epoch_start
        print("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}"
                .format(epoch, ellapsed_time, psnr))
            
        with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
            f.write("Epoch: {:03d}| time: {:.0f}| PSNR: {:.3f}"
                .format(epoch, ellapsed_time, psnr))
            f.write('\n')
        
        save_path = manager.save()


if config.train_bijective:

    print('Training of the Bijective Subnetwork:')
    print('---> Dataset: {}'.format(config.dataset))
    print('---> Experiment path: {}'.format(exp_path))
    print('---> Bijective depth: {}'.format(config.bij_depth))
    print('---> Num epochs: {}'.format(config.n_epochs_bij))
    print('---> Learning rate: {}'.format(config.lr))

    ngrid = int(np.sqrt(config.n_test))
    image_path_generated = os.path.join(samples_folder, 'Generated samples')
    os.makedirs(image_path_generated, exist_ok=True)

    z_inters = np.zeros([len(list(train_dataset)) * config.batch_size , latent_dim])
    cnt = 0
    for x in train_dataset:
        z_inter, _ = inj_model(x, reverse = False)
        z_inters[cnt*config.batch_size:(cnt+1)*config.batch_size] = z_inter.numpy()
        cnt = cnt + 1

    z_inters = tf.convert_to_tensor(z_inters, tf.float32)
    z_inters_dataset = tf.data.Dataset.from_tensor_slices((z_inters))
    z_inters_dataset = z_inters_dataset.shuffle(config.batch_size * 3).batch(config.batch_size , drop_remainder = True).prefetch(5)
            
    for epoch in range(config.n_epochs_bij):
        epoch_start = time()
        for x in z_inters_dataset:
            ml_loss = train_step_ml(x, bij_model, pz, optimizer_bij).numpy()
                      
        # Sampling
        z_base = pz.prior.sample(config.n_test) # sampling from base (Gaussian) 
        z_inter = bij_model(z_base , reverse = True)[0] # Intermediate samples 
        generated_samples = inj_model(z_inter , reverse = True)[0].numpy() # Randmly generated samples
        
        generated_samples = generated_samples[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        generated_samples = generated_samples.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_generated, '%d_samples.png' % (epoch,)),
            generated_samples[:,:,0]) # Generated samples

        epoch_end = time()       
        ellapsed_time = epoch_end - epoch_start
        print("Epoch: {:03d}| time: {:.0f}| ML Loss: {:.3f}"
                .format(epoch, ellapsed_time, ml_loss))
            
        with open(os.path.join(exp_path, 'results.txt'), 'a') as f:
            f.write("Epoch: {:03d}| time: {:.0f}| ML Loss: {:.3f}"
                .format(epoch, ellapsed_time, ml_loss))
            f.write('\n')
        
        save_path = manager.save()


if config.inverse_scattering_solver:

    print('Solving Inverse Scattering:')
    print('---> Dataset: {}'.format(config.dataset))
    print('---> Experiment path: {}'.format(exp_path))
    print('---> Epsilon_r: {}'.format(config.er))
    print('---> Noise snr: {}'.format(config.noise_snr))
    print('---> Solver:{}'.format(config.solver))
    print('---> Initial guess: {}'.format(config.initial_guess))
    print('---> Optimizer: {}'.format(config.optimizer))
    print('---> Solver learning rate:{}'.format(config.lr_inv))

    n_test = 1 if config.scattering_data == 'real' else config.num_objects
    testing_images = next(iter(test_dataset))[:n_test]
    scattering_op = scattering_utils.scattering_op(n_inc_wave = 12)

    scattering_pipeline = scattering_utils.scattering_solver(exp_path, scattering_op, inj_model, bij_model, pz= pz)

    if config.scattering_data == 'real':
        if config.fresnel_sample == 'FoamDielExt':
            setup = np.load('scattering_config/FoamDielExt.npz')
        elif config.fresnel_sample == 'FoamTwinDiel':
            setup = np.load('scattering_config/FoamTwinDiel.npz')

        testing_images = setup['gt']
        testing_images = cv2.resize(testing_images , (config.img_size,config.img_size))
        testing_images = testing_images[None,...][...,None]
        testing_images_write = (config.er-1) * ((testing_images + 1)/2) + 1

        gt_path = os.path.join(scattering_pipeline.prob_folder, 'gt.png')
        plt.imshow(testing_images_write[0,:,:,0], cmap = config.cmap)
        plt.colorbar()
        plt.savefig(gt_path)
        plt.close()
        testing_images = tf.convert_to_tensor(testing_images)
        Es = setup['Es']
        measurements = tf.convert_to_tensor(Es[None,...], tf.complex64)

    else:
        measurements = scattering_pipeline.forward_solver(testing_images)

    if config.run_map:
        if config.solver == 'lso':
            MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images , lam=0) 
        elif config.solver == 'dso':
            MAP_estimate = scattering_pipeline.MAP_estimator(measurements, testing_images, lam=1e-2) 

    if config.run_posterior:
        scattering_pipeline.posterior_sampling(measurements[config.test_nb:config.test_nb+1], testing_images[config.test_nb:config.test_nb+1])

        
        