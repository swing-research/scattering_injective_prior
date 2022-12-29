import tensorflow as tf
import numpy as np
import argparse
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

        
def data_normalization(x):
    
    x = x.astype('float32')
    x = x - (x.max() + x.min())/2
    x /= (x.max())
    
    return x

def image_resizer(x , r):
    b , h, _ , nch = np.shape(x)
    y = np.zeros((np.shape(x)[0], r, r, nch))
    
    if x.shape[1] != r:
        for i in range(b):
            if nch == 1:
                y[i,:,:,0] = cv2.resize(x[i] , (r,r))
            else:
                y[i] = cv2.resize(x[i] , (r,r))
                
    else:
        y = x
        
    return y
 
def PSNR(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += psnr(x_pred[i],
             x_true[i],
             data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]

def SSIM(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += ssim(x_pred[i],
             x_true[i],
             data_range=x_true[i].max() - x_true[i].min(),
             multichannel=True)
        
    return s/np.shape(x_pred)[0]



def Dataset_preprocessing(dataset = 'MNIST', batch_size = 64):
    
    if dataset == 'mnist':
        r = 32
        (train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
        
        train_images = np.expand_dims(train_images, axis = 3)
        test_images = np.expand_dims(test_images, axis = 3)
        
    elif dataset == 'ellipses':
        
        images = np.load('/raid/Amir/Projects/datasets/ellipses/ellipses_diverse_64.npy')
        train_images , test_images = np.split(images , [55000])
     
        r = 64    

        
    train_images = image_resizer(train_images, r)
    test_images = image_resizer(test_images, r)
    train_images = data_normalization(train_images)
    test_images = data_normalization(test_images)
    
    train_images = data_normalization(train_images)
    test_images = data_normalization(test_images)
    
    train_images = tf.convert_to_tensor(train_images, tf.float32)
    test_images = tf.convert_to_tensor(test_images, tf.float32)
   
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images))
    
    SHUFFLE_BUFFER_SIZE = 256
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(batch_size,
                                                                     drop_remainder = True).prefetch(5)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset , test_dataset
      
  

    
def flags():

    parser = argparse.ArgumentParser()
     
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=200,
        help='number of epochs to train for')
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='batch_size')

   
    parser.add_argument(
        '--dataset', 
        type=str,
        default='mnist',
        help='mnist or ellipses')
    
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate')
    
    
    parser.add_argument(
        '--ml_threshold', 
        type=int,
        default= 100,
        help='when should ml training begin')


    parser.add_argument(
        '--injective_depth',
        type=int,
        default= 3,
        help='revnet depth of injective sub-network')
    
    parser.add_argument(
        '--bijective_depth',
        type=int,
        default= 2,
        help='revnet depth of bijective sub-network')
    
        
    parser.add_argument(
        '--gpu_num',
        type=int,
        default=0,
        help='GPU number')

    
    parser.add_argument(
        '--desc',
        type=str,
        default='Default',
        help='add a small descriptor to the experiment')
    
    parser.add_argument(
        '--train',
        type=int,
        default= 1,
        help='Training or just load')

    parser.add_argument(
        '--reload',
        type=int,
        default= 0,
        help='reload the existing model if exists')

    ######################################################################
    # For solving scattering
    
    parser.add_argument(
        '--inv',
        type=int,
        default= 0,
        help='Running inverse scattering solver')
    
    parser.add_argument(
        '--noise_snr',
        type=float,
        default=30,
        help='Noise SNR (dB)')

    parser.add_argument(
        '--initial_guess',
        type=str,
        default='MOG',
        help='Initial guess: BP or MOG')

    parser.add_argument(
        '--er',
        type= float,
        default= 3,
        help='Maximum epsilon_r of the medium')

    parser.add_argument(
        '--nsteps',
        type= int,
        default= 500,
        help='Number of steps for solver optimization')
    parser.add_argument(
        '--optimization_mode',
        type=str,
        default='latent_space',
        help='Choose where to apply optimization, latent_space or data_space')
    
    parser.add_argument(
        '--lr_inv',
        type=float,
        default= 1e-2,
        help='Learning rate of inverse scattering solver')
    
    
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed
