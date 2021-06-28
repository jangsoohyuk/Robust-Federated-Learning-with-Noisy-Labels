# Robust Federated Learning with Noisy Labels
This is an unofficial PyTorch implementation of [Robust Federated Learning with Noisy Labels](https://arxiv.org/abs/2012.01700). 

## Requirements
- python 3.8.8
- pytorch 1.8.0
- torchvision 0.9.0 

## Usage
Results can be reproduced running the following:

#### MNIST

```
python3 main.py --gpu 0 --iid --dataset mnist --epochs 1000 --noise_type symmetric --noise_rate 0.2 
```
```
python3 main.py --gpu 0 --iid --dataset mnist --epochs 1000 --noise_type pairflip --noise_rate 0.2 
```

#### CIFAR10

```
python3 main.py --gpu 0 --iid --dataset cifar --epochs 1000 --noise_type symmetric --noise_rate 0.2 
```
```
python3 main.py --gpu 0 --iid --dataset cifar --epochs 1000 --noise_type pairflip --noise_rate 0.2 
```

## References
- Yang, S., Park, H., Byun, J., & Kim, C. (2020). Robust Federated Learning with Noisy Labels. arXiv preprint arXiv:2012.01700.

# Acknowledgements
This codebase was adapted from https://github.com/shaoxiongji/federated-learning and https://github.com/bhanML/Co-teaching