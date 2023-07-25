# Deep Injective Prior for Inverse Scattering
[![Paper](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2301.03092)
[![PWC](https://img.shields.io/badge/PWC-report-blue)](https://paperswithcode.com/paper/deep-injective-prior-for-inverse-scattering)

This repository is the official Tensorflow Python implementation of "[Deep Injective Prior for Inverse Scattering](https://arxiv.org/abs/2301.03092)".

| [**Project Page**](https://sada.dmi.unibas.ch/en/research/injective-flows)  | 


<p float="center">
<img src="https://github.com/swing-research/scattering_injective_prior/blob/main/figures/network.jpg" width="800">
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
You should specify the training parameters in config.py and run the following command:
```sh
python3 main.py
```

### Solving inverse scattering problem using the injective generator as prior:
You should download the scattering configuration files from [here](https://drive.switch.ch/index.php/s/42ySXbEyakP9vQd) and put "scattering_config" folder in the project directory.
You should specify the parameters for MAP estimation and posterior sampling in config.py and run the following command:
```sh
python3 main.py
```

<p float="center">
<img src="https://github.com/swing-research/scattering_injective_prior/blob/main/figures/posterior_real_32.jpg" width="800">
</p>

## Citation
If you find the code or our dataset useful in your research, please consider citing the paper.

```
@article{khorashadizadeh2022deepinjective,
  title={Deep Injective Prior for Inverse Scattering},
  author={Khorashadizadeh, AmirEhsan and Khorashadizadeh, Vahid and Vandenbosch Guy A.E. and Eskandari, Sepehr and Dokmani{\'c}, Ivan},
  journal={arXiv preprint arXiv:2301.03092},
  year={2023}
}
```
