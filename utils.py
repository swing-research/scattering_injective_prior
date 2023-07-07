import tensorflow as tf
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import config
from tqdm import tqdm
from matplotlib.patches import Ellipse, Circle
import matplotlib.pyplot as plt
import numpy.random as rnd
import os


def fantom_generation(n_samples = 60000 ,
                       num_fantom_max = 4 ,
                       generation = True):
    
    if generation:
        
        with tqdm(total=n_samples) as pbar: 
            for k in range(n_samples):
                
                num_fantom = np.random.randint(1,num_fantom_max)
                fantoms = [Circle(rnd.uniform(3,7 , 2),
                                  rnd.uniform(1,4))
                        for i in range(num_fantom)]
                
                fig = plt.figure(figsize = ( 4, 4) , dpi = 64)
                ax = fig.add_subplot(111 , frameon=False)
                for e in fantoms:
                    ax.add_artist(e)
                    e.set_clip_box(ax.bbox)
                    e.set_alpha(np.random.uniform(0.3 , 1))
                    e.set_facecolor([0,0,0])
                
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
                
                plt.show()
                fig.savefig('datasets/circles_256/{}.png'.format(k))
                plt.close()
    
                pbar.set_description('generating...')
                pbar.update(1)
                
    else:
        
        folder = 'datasets/circles_256/'
        image_names = os.listdir(folder)
        r = 64
        n_samples = len(image_names)
        
        x = np.zeros([n_samples,r,r,1])
        
        with tqdm(total=n_samples) as pbar: 
            for i in range(len(image_names)):
                
                image_add = folder + image_names[i]
                image = cv2.imread(image_add)
                if image is None:
                    continue
                image = 255.0 - image
                x[i , :, :, 0] = cv2.resize(image , (r,r))[:,:,0]
                
                pbar.set_description('processing...')
                pbar.update(1)
        
        np.save('datasets/circles_64.npy' , x)

@tf.function
def train_step_mse(sample, inj_model, optimizer_inj):
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
def train_step_ml(sample, bij_model, pz, optimizer_bij):
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
        s += psnr(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min())
        
    return s/np.shape(x_pred)[0]

def SSIM(x_true , x_pred):
    
    s = 0
    for i in range(np.shape(x_pred)[0]):
        s += ssim(x_true[i],
                  x_pred[i],
                  data_range=x_true[i].max() - x_true[i].min(),
                  multichannel=True)
        
    return s/np.shape(x_pred)[0]



def Dataset_preprocessing(dataset = 'MNIST', batch_size = 64):
    
    if dataset == 'mnist':

        (train_images, train_labels), (test_images, _) = tf.keras.datasets.mnist.load_data()
        if config.ood_analysis:

            np.random.seed(0)

            sorted_labels_ind = np.argsort(train_labels)
            sorted_labels = train_labels[sorted_labels_ind]
            test_ind = np.where(sorted_labels == 6)[0][0]

            train_images = train_images[sorted_labels_ind,:,:]
            test_images = train_images[test_ind:]
            train_images = train_images[:test_ind]
            np.random.shuffle(test_images)
            np.random.shuffle(train_images)



        train_images = np.expand_dims(train_images, axis = 3)
        test_images = np.expand_dims(test_images, axis = 3)
        
    elif dataset == 'ellipses':
        
        images = np.load('datasets/ellipses_64.npy')
        train_images , test_images = np.split(images , [55000])

    elif dataset == 'circles':
        
        images = np.load('datasets/circles_64.npy')
        train_images , test_images = np.split(images , [55000])

    r = config.img_size
  
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
      
  

if __name__ == '__main__':
    fantom_generation(n_samples=60000, num_fantom_max=4, generation= False)