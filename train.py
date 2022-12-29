import numpy as np
from time import time
import tensorflow as tf
from utils import *
from models import injective, bijective, prior
import scattering_solver
import imageio
import os

FLAGS, unparsed = flags()
num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
dataset = FLAGS.dataset
lr = FLAGS.lr
gpu_num = FLAGS.gpu_num
desc = FLAGS.desc
ml_threshold = FLAGS.ml_threshold
injective_depth = FLAGS.injective_depth
bijective_depth = FLAGS.bijective_depth
reload = bool(FLAGS.reload)

initial_guess = FLAGS.initial_guess
lr_inv = FLAGS.lr_inv
run_train = bool(FLAGS.train)
run_inv = bool(FLAGS.inv)
noise_snr = FLAGS.noise_snr
er = FLAGS.er
optimization_mode = FLAGS.optimization_mode
nsteps = FLAGS.nsteps

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


# Print the experiment setup:
print('Experiment setup:')
print('---> num_epochs: {}'.format(num_epochs))
print('---> batch_size: {}'.format(batch_size))
print('---> dataset: {}'.format(dataset))
print('---> Learning rate: {}'.format(lr))
print('---> experiment path: {}'.format(exp_path))
print('---> epsilon_r: {}'.format(er))
print('---> noise_snr: {}'.format(noise_snr))
print('---> initial guess: {}'.format(initial_guess))
print('---> learning rate of inverse problem solver:{}'.format(lr_inv))
print('---> optimization mode:{}'.format(optimization_mode))

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


# call generator once to set weights (Data dependent initialization)
dummy_x = next(iter(train_dataset))
dummy_z, _ = inj_model(dummy_x, reverse=False)
dummy_l_z , _ = bij_model(dummy_z, reverse=False)


ckpt = tf.train.Checkpoint(pz = pz , inj_model=inj_model,optimizer_inj=optimizer_inj,
    bij_model=bij_model,optimizer_bij= optimizer_bij)
manager = tf.train.CheckpointManager(
    ckpt, os.path.join(exp_path, 'checkpoints'), max_to_keep=5)

if reload:
    ckpt.restore(manager.latest_checkpoint)

@tf.function
def train_step_mse(sample):
    """MSE training of the injective sub-network"""

    with tf.GradientTape() as tape:
        
        MSE = tf.keras.losses.MeanSquaredError()
        z , _ = inj_model(sample, reverse= False)
        recon = inj_model(z , reverse = True)[0]
        mse_loss = MSE(sample , recon)
        loss = mse_loss
        
        variables= tape.watched_variables()
        grads = tape.gradient(loss, variables)
        optimizer_inj.apply_gradients(zip(grads, variables))

    return loss



@tf.function
def train_step_ml(sample):
    """ML training of the bijective sub-network"""

    with tf.GradientTape() as tape:
        latent_sample, obj = bij_model(sample, reverse=False)
        p = -tf.reduce_mean(pz.prior.log_prob(latent_sample))
        j = -tf.reduce_mean(obj) # Log-det of Jacobian
        loss =  p + j
        variables = tape.watched_variables()
        grads = tape.gradient(loss, variables)
        optimizer_bij.apply_gradients(zip(grads, variables))

    return loss

if manager.latest_checkpoint and reload:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")

if run_train:
    z_inters = np.zeros([len(list(train_dataset)) * batch_size , latent_dim])
    for epoch in range(num_epochs):
        epoch_start = time()
        if epoch < ml_threshold:
            # MSE traiing of the injective network for ml-threshold epochs
            for x in train_dataset:
                train_step_mse(x)
                
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
                ml_loss = train_step_ml(x).numpy()
                
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
            print('Number of trainable_parameters of injective model: {}'.format(parameters_inj_model))
            print('Number of trainable_parameters of bijective model: {}'.format(parameters_bij_model))
            print('Total number of trainable_parameters: {}'.format(parameters_inj_model + parameters_bij_model))
    

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
        imageio.imwrite(os.path.join(image_path_reconstructions, 'recon_%d.png' % (epoch,)),
            test_recon) # Reconstructed test images
        
        test_gt = test_gt.numpy()[:, :, :, ::-1].reshape(ngrid, ngrid,image_size,
            image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1, c)* 127.5 + 127.5
        test_gt = test_gt.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_reconstructions, 'gt_%d.png' % (epoch,)),
            test_gt) # Ground truth test images
        
        generated_samples = generated_samples[:, :, :, ::-1].reshape(ngrid, ngrid,
            image_size, image_size, c).swapaxes(1, 2).reshape(ngrid*image_size, -1,
            c)*127.5 + 127.5
        generated_samples = generated_samples.clip(0, 255).astype(np.uint8)
        imageio.imwrite(os.path.join(image_path_generated, 'samples_%d.png' % (epoch,)),
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

if run_inv:
    
    testing_images = next(iter(test_dataset))
    operator = scattering_solver.Inverse_scattering(n_inc_wave = 12 , er = er,
        image_size = image_size)
    scattering_solver.solver(testing_images[:25], exp_path,
                                operator, noise_snr, inj_model, bij_model,
                                latent_dim,pz= pz,
                                initial_guess = initial_guess,
                                nsteps = nsteps,
                                optimization_mode = optimization_mode,
                                er = er,
                                lr_inv = lr_inv)