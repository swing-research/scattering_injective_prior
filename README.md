# Deep Injective Prior for Inverse Scattering
[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2301.03092)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/deep-injective-prior-for-inverse-scattering)

This repository is the official Tensorflow Python implementation of "[Deep Injective Prior for Inverse Scattering](https://arxiv.org/abs/2301.03092)".

| [**Project Page**](https://sada.dmi.unibas.ch/en/research/injective-flows)  | 


<p float="center">
<img src="https://github.com/swing-research/scattering_injective_prior/blob/main/figures/network.png" width="800">
</p>


## Requirements
(This code is tested with tensorflow-gpu 2.3.0, Python 3.8.3, CUDA 11.0 and cuDNN 7.)
- numpy
- scipy
- matplotlib
- sklearn
- opencv-python
- tensorflow-gpu==2.3.0
- tensorflow-probability==0.11.1


## Experiments
### Training the injective generative model:
We used MNIST and the custom ellipses datasets. You can download the ellipses dataset from [here](https://drive.switch.ch/index.php/s/yFGPLw2pAsNTkkj), unzip the file, and put the .npy file in the datasets/ellipses/. 
This is an example of how training the model for 300 epochs (150 for the injective part (with revnet depth 3) and 150 for the bijective part (with revent depth 4)), over the ellipses dataset.
```sh
python3 train.py --train 1 --num_epochs 300 --ml_threshold 150 --injective_depth 3 --bijective_depth 4 --dataset ellipses --gpu_num 0 --desc default
```
Each argument is explained in detail in utils.py.

### Solving inverse scattering problem using the injective generator as prior:
You should download the scattering configuration files for  [64x64](https://drive.switch.ch/index.php/s/6HOH8PN8BonwR5W) and [32x32](https://drive.switch.ch/index.php/s/51A2ZvFLd2NI5Bj) resolutions and put the .npz files in the folder scattering_config/.
As soon as the network is trained, you can reload it to be used as a prior for solving inverse scattering for 500 iterations with learning rate 0.05 with this configuration: 30dB noise, MOG initialization, epsilon_r= 4 and optimizing over latent space:
```sh
python3 train.py --reload 1 --train 0 --inv 1 --injective_depth 3 --bijective_depth 4 --dataset ellipses --gpu_num 0 --desc default  --nsteps 500 --lr_inv 0.05 --noise_snr 30 --initial_guess MOG --er 4  --optimization_mode latent_space
```

## Citation
If you find the code or our dataset useful in your research, please consider citing the paper.

```
@article{khorashadizadeh2022deepinjective,
  title={Deep Injective Prior for Inverse Scattering},
  author={Khorashadizadeh, AmirEhsan and Eskandari, Sepehr and Khorashadi-Zadeh, Vahid and Dokmani{\'c}, Ivan},
  journal={arXiv preprint arXiv:2301.03092},
  year={2023}
}
```
