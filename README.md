# mmengine-bayes
This repo is a fork of [openmmlab-mmengine](https://github.com/open-mmlab/mmengine). It adds support for training and evaluating **Laplace-approximated Bayesian models** for **pre-trained openmmlab models**.

Read the original [README](https://github.com/open-mmlab/mmengine/blob/main/README.md). 

## 🚀 New Features 

- **New loop objects**: Loop objects `FisherLoop` and `TestUncLoop` added in `mmengine/runner/loops.py`, iterators for training KFAC-fisher and testing predictive uncertainty
- **New runner functions**: added functionalities to `mmengine/runner/runner.py`


<!-- 👉 **Try it out**: [Live Demo Link] (if applicable)   -->

## Installation

```bash
git clone https://github.com/romiebanerjee/mmengine-bayes
cd mmengine-bayes
pip install -e .
```




