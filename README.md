# Robust-Federated-Learning-with-Noisy-Labels
This is an unofficial PyTorch implementation of [Robust Federated Learning with Noisy Labels](https://arxiv.org/abs/2012.01700). 

## Requirements
- python 3.8.8
- pytorch 1.8.0
- torchvision 0.9.0 

## Usage
Train the network on the Symmmetric Noise CIFAR-10 dataset (noise rate = 0.2):

```
python3 main.py --gpu 0 --iid --dataset cifar --noise_type symmetric --noise_rate 0.2 
```

## References
- 