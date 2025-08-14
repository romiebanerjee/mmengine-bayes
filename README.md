# mmengine-bayes
This repo is a fork of [open-mmlab mmengine](https://github.com/open-mmlab/mmengine). It adds support for training and evaluating **Laplace-approximated Bayesian models** for **pre-trained openmmlab models**.

Read the original [README](https://github.com/open-mmlab/mmengine/blob/main/README.md). 

## 🚀 New Features 

- **Curvature objects**: New object class `mmengine.curvature.KFAC` supporting kronecker-factored Laplace posterior of a trained model 
- **Gradient objects**: The `KFAC` objects supports kronecker-factored gradients and inner products  
- **Loop objects**: Loop objects `FisherLoop` and `TestUncLoop` added in `mmengine/runner/loops.py`, iterators for training KFAC-fisher and testing predictive uncertainty
- **New runner functions**: added functionalities to `mmengine/runner/runner.py`
- **Model-agnostic**: works with any OpenMMlab model

KFAC code adapted from [https://github.com/DLR-RM/curvature](https://github.com/DLR-RM/curvature)

## Installation

```bash
git clone https://github.com/romiebanerjee/mmengine-bayes
cd mmengine-bayes
pip install -e .
```

## Usage

### Estimate Fisher of a pre-trained model
- *Input*: `model` a pre-trained OpenMMlab (Pytorch) model with weights dict `weights` 
- *KFAC*: Instantiate a `mmengine.curvature.KFAC` object with `model` and `weights`
- *Iterator*: The `FisherLoop` iterator (similar to the `TrainLoop` iterator) 
    - runs forward-pass on the model `model` loaded with weights `weights`
    - through the training dataset (batch size = 1), without `optimizer.step()` (unlike )
    - At every batch iteration, the run `KFAC.update_fisher()` to update `KFAC.state` dictionary 
     - Run one epoch 
- *method*: `runner.fisher()`

### Calculate Uncertainty
- *Input*: `model`:model, `weights`:model_weights_dict, `kfac`:model_kfac, `kfac.state`:kf-fisher_dict, `val_loader`
- *Iterator*: The `TestUncLoop` will iteratively calculate downstream **GLM** predictive distribution covariance, through the validation dataset, using the following methods:
    - `MC-GLM`: A monte-carlo estimator for GLM predictive covariance
    - `1-GLM`: Rank-One GLM predictive covariance, using single monte-carlo sample
    - `e-GLM`: Rank-One GLM predictive, using a single fisher eigen-direction in weight space 
- *method*: `runner.test_unc()`

## Theory
- [Laplace Bayesian Neural Networks Tutorial](docs/en/advanced_tutorials/laplace-bnn-math.md)