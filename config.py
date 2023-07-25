# Training of the generative model
train_injective = True # Train the injective subnetwork
train_bijective = True #  Train the bijective subnetwork
n_epochs_inj = 150 # Number of epochs for injective subnetwork
n_epochs_bij = 150 # Number of epochs for bijective subnetwork
img_size = 32 # Image size
batch_size = 64
dataset = 'mnist' # 'mnist' or 'ellipses' dataset
lr = 1e-4 # learning rate for the training of the networks
gpu_num = 0 # Gpu selection
desc = 'default' # Add a small descriptor to the experiment
inj_depth = 3 # Injective network depth
bij_depth = 2 # Bijective network depth
reload = True # Reload the existing trained network if exists
ood_experiment = False # Out-of-distribution experiment of the paper
unet_coupling = True # U-Net as coupling layers is more powerful but with slow training
n_test = 25 # Number of test samples to show in the results

inverse_scattering_solver = True # Running inverse scattering solver
# MAP estimation
run_map = True # Run the MAP solver
reload_solver = True # Reload the existing solver if exists
problem_desc = 'default' # Add a small descriptor to the experiment
noise_snr = 30 # Noise SNR (dB) added to the measurements of scattered fields
er = 2.0 # Maximum epsilon_r of the medium
solver = 'lso' # 'lso' or 'dso'
optimizer = 'Adam' # 'Adam' or 'lbfgs'
lr_inv = 5e-2 # Learning rate of inverse scattering solver for Adam
initial_guess = 'MOG' # Initial guess: 'BP' or 'MOG'
nsteps = 300 # Number of steps for optimizer
scattering_data = 'synthetic' # 'synthetic' or 'real'
fresnel_sample = 'FoamDielExt' # just in case of real data: 'FoamDielExt' or 'FoamTwinDiel'
cmap = 'seismic' # Color map
tv_weight = 0.00 # TV multiplier
num_objects = 1 # Number of test samples to be used in inverse scattering solver
# Warning: You can use lbfgs only for solving the problem over a single image
# So please set num_objects = 1 for lbfgs solver.


# Posterior modeling
run_posterior = True # Run the posterior sampling
reload_posterior = False # Reload the existing posterior model if exists
nsteps_posterior = 10000 # Number of steps for optimization
lr_posterior = 1e-2 # Learning rate of optimizer
test_nb = 0 # Posterior can only be run over a single image, which sample from the test data
beta = 0.01 # The KL multiplier