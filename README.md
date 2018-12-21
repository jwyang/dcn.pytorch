# Introduction

This repo is a pytorch implementation of [Deformable Convolution Networks](https://arxiv.org/abs/1703.06211). It is ported from a previous pytorch [implemetnation](https://github.com/1zb/deformable-convolution-pytorch), which is transformed from original MXNet [implementation](https://github.com/msracver/Deformable-ConvNets).

## What's the difference from other implementations?

This repo supports Pytorch 1.0, which uses a more convinient C++ and Cuda exntesion tools [ATen](https://github.com/zdevito/ATen). 

## Todo list

- [ ] Found a minor bug in pytorch 1.0 version (do not use 1.0 for now).
- [ ] Benchmark the performance on object detection ([faster r-cnn](https://github.com/jwyang/faster-rcnn.pytorch)).

## Compilation

### Prerequiestes

* Python 2.7 or 3.6
* Pytorch 0.4 or Pytorch 1.0
* CUDA 8.0 or higher

### Build Pytorch-0.4 version

```
sh make.sh
CC=g++ python build.py
```

### Build Pytorch-1.0 version

```
python setup.py build develop
```

See `test.py` for example usage.

### Notice
Only `torch.cuda.FloatTensor` is supported.


