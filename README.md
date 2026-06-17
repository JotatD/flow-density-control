# Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning

[![arXiv](http://img.shields.io/badge/arxiv-2511.22640-red?logo=arxiv)](https://www.arxiv.org/abs/2511.22640)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riccardodesanti/flow-density-control/blob/main/tutorial.ipynb)

This repository contains the official implementation of the Flow Density Control (FDC) algorithm, a method for optimizing general functionals of the generated distribution — including entropy, diversity, and coverage objectives — beyond what entropy-regularized reward maximization can achieve.

## Installation

Check out the repo and `cd` into it:

```bash
git clone https://github.com/riccardodesanti/flow-density-control && cd flow-density-control
```

Then to build the environment either use pip:

```bash
pip install torch==2.3.* dgl==2.4 --find-links https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install -e .
```

Or (recommended) first install `uv` here: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
Then run:

```bash
uv sync
```

You can also install directly from GitHub without cloning:

```bash
pip install git+https://github.com/riccardodesanti/flow-density-control
```

## Overview

FDC is built on top of [diffusiongym](https://github.com/cristianpjensen/diffusiongym), a library for reward adaptation of pre-trained flow models across any data modality. To run FDC on your own model you need three things:

1. A **data type** (e.g. `DDTensor` for plain tensors, or a custom `DDMixin` subclass for structured data)
2. A **base model** (`BaseModel[D]`) wrapping your pre-trained network
3. A **reward** (`Reward[D]`) measuring the quality of generated samples

diffusiongym then handles environment construction, SDE simulation, and trajectory storage. `FDCTrainer` runs the mirror-descent optimization loop on top using adjoint matching.

## Quickstart

Check `tutorial.ipynb` for a complete worked example on a toy 1D trimodal GMM:

```python
import torch, diffusiongym
from omegaconf import OmegaConf
from genexp.trainers.fdc import FDCTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = diffusiongym.make(
    base_model="1d/trimodal_gmm",
    reward="1d/sigmoidal",
    discretization_steps=50,
    device=device,
)

config = OmegaConf.load("configs/example_fdc.yaml")
# or equivalently:
config = OmegaConf.create({
    "gamma": 0.32, "beta": 0.2, "epsilon": 0.005,
    "num_md_iterations": 3,
    "adjoint_matching": {
        "lr": 5e-4, "batch_size": 128, "num_iterations": 2,
        "finetune_steps": 50, "sampling": {"num_samples": 512},
    },
})

trainer = FDCTrainer(config, env, device=device)
losses = trainer.fit(config.num_md_iterations)
```

## Usage

### 1. Data type

For plain tensor data, use the built-in `DDTensor`:

```python
from diffusiongym import DDTensor

x = DDTensor(torch.randn(batch_size, dim))
```

For structured data (graphs, molecules, images with conditioning), use one of the existing types or subclass `DDMixin` and implement `apply`, `combine`, `aggregate`, `collate`, `__len__`, and `__getitem__`. See [diffusiongym's types documentation](https://cristianpjensen.github.io/diffusiongym/) for details.

### 2. Base model

Subclass `BaseModel[D]` and set `output_type` to one of `"velocity"`, `"score"`, `"epsilon"`, or `"endpoint"` depending on what your network predicts:

```python
import torch
import torch.nn as nn
from typing import Any
from diffusiongym import BaseModel, DDTensor, OptimalTransportScheduler
from diffusiongym.schedulers import Scheduler

class MyFlowModel(BaseModel[DDTensor]):
    output_type = "velocity"   # or "score" | "epsilon" | "endpoint"

    def __init__(self, dim: int, device=None):
        super().__init__(device)
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 256), nn.SiLU(), nn.Linear(256, dim)
        )
        self._scheduler = OptimalTransportScheduler()

    @property
    def scheduler(self) -> Scheduler[DDTensor]:
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict]:
        return DDTensor(torch.randn(n, dim, device=self.device)), kwargs

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        t_in = t.unsqueeze(1) if t.ndim == 1 else t
        out = self.net(torch.cat([x.data, t_in], dim=1))
        return DDTensor(out)
```

`FDCTrainer` converts all outputs to a score function internally using the scheduler. The `scheduler` defines the interpolant `x_t = α_t x_1 + β_t x_0`. `OptimalTransportScheduler` uses the linear schedule `α_t = t`, `β_t = 1 − t`. `CosineScheduler` and `DiffusionScheduler` are also available.

The base class provides `train_loss(x1)` automatically once `output_type` and `scheduler` are set, so you can train your model with:

```python
import diffusiongym
diffusiongym.train_base_model(model, optimizer, data, steps=10_000)
```

### 3. Reward

`FDCTrainer` accepts any `Reward[D]` from diffusiongym:

```python
from diffusiongym.rewards import Reward
from diffusiongym.types import DDTensor

class MyReward(Reward[DDTensor]):
    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs):
        return some_score(sample.data)   # scalar or tensor in [0, 1]
```

### 4. Environment

Create an environment that matches your model's `output_type`. The simplest way is `construct_env`, which picks the right environment class automatically:

```python
import diffusiongym

env = diffusiongym.construct_env(
    base_model=model,
    reward=MyReward(),
    discretization_steps=100,
    reward_scale=1.0,
)
```

If your model and rewards are registered in diffusiongym's registries you can also use the `make()` factory:

```python
from diffusiongym import base_model_registry

@base_model_registry.register("mytask/mymodel")
class MyFlowModel(BaseModel[DDTensor]):
    ...

env = diffusiongym.make(
    base_model="mytask/mymodel",
    reward="mytask/myreward",
    discretization_steps=100,
    device=device,
)
```

Alternatively, instantiate the environment class directly:

```python
from diffusiongym import VelocityEnvironment   # or Score/Epsilon/EndpointEnvironment

env = VelocityEnvironment(model, MyReward(), discretization_steps=100)
```

### 5. FDC trainer

Pass the environment directly to `FDCTrainer`. The trainer creates its own deep copies of `env.base_model` for the fine and reference models:

```python
from omegaconf import OmegaConf
from genexp.trainers.fdc import FDCTrainer

config = OmegaConf.create({
    "gamma": 0.32,      # score-weighting strength
    "beta": 0.0,        # KL subtraction coefficient
    "epsilon": 0.01,    # clipping for t → 1
    "adjoint_matching": {
        "lr": 1e-4,
        "batch_size": 128,
        "num_iterations": 2,    # AM rounds per mirror-descent step
        "finetune_steps": 50,
        "sampling": {"num_samples": 512},
    },
})

trainer = FDCTrainer(config, env, device=device)
```

Then run the mirror-descent loop with `fit`:

```python
losses = trainer.fit(num_iterations=10)
```

`fit` returns a flat list of per-round adjoint matching losses.

Each iteration performs one **expand** step: adjoint matching driven by the score function of the current base model as the reward signal, which steers the fine-tuned model toward regions of higher density under the evolving reference. After each iteration the base model is updated to the fine-tuned model, implementing the mirror-descent update.

## Citation

If you use this code in your research, please include the following citations in your work:

```
@inproceedings{de2025flow,
      title={Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning}, 
      author={Riccardo De Santi and Marin Vlastelica and Ya-Ping Hsieh and Zebang Shen and Niao He and Andreas Krause},
      year={2025},
      booktitle={Advances in Neural Information Processing Systems},
      url={https://arxiv.org/abs/2511.22640}, 
}

@inproceedings{de2025provable,
      title={Provable Maximum Entropy Manifold Exploration via Diffusion Models}, 
      author={Riccardo De Santi and Marin Vlastelica and Ya-Ping Hsieh and Zebang Shen and Niao He and Andreas Krause},
      year={2025},
      booktitle={Proceedings of the 42nd International Conference on Machine Learning},
      url={https://arxiv.org/abs/2506.15385}, 
}
```

Reference for the [diffusiongym](https://github.com/cristianpjensen/diffusiongym) library:

```
@inproceedings{jensen2026value,
  title={Value Matching: Scalable and Gradient-Free Reward-Guided Flow Adaptation},
  author={Cristian Perez Jensen and Luca Schaufelberger and Riccardo De Santi and Kjell Jorner and Andreas Krause},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
}
```
