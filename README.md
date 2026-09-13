# BloodMNIST Generative Models

An academic exploration of generative deep-learning approaches for BloodMNIST images. The
repository places four independently developed experiments in one reproducible, privacy-aware
workspace:

- conditional GAN (cGAN);
- DCGAN;
- denoising diffusion probabilistic model (DDPM);
- convolutional variational autoencoder (VAE).

## Scope

The aim was to compare distinct generative-model families on a small medical-image benchmark.
The notebooks cover model construction, training loops, image generation and FID-oriented
evaluation. The VAE implementation is also available as a reusable Python module.

This is a cleaned portfolio edition of an academic team project. The original dataset, generated
samples, model checkpoints, TensorBoard logs and internal report are deliberately excluded.

## Repository layout

```text
.
├── notebooks/
│   ├── CGAN.ipynb
│   ├── DCGAN.ipynb
│   ├── DDPM.ipynb
│   └── VAE_bloodmnist_testing.ipynb
├── src/
│   ├── __init__.py
│   └── vae_bloodmnist.py
└── requirements.txt
```

## Setup

Use Python 3.10 or newer in an isolated virtual environment.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch Jupyter from the repository root so that the local `src` package is available:

```bash
jupyter lab
```

Open one notebook at a time from `notebooks/`. The BloodMNIST files are downloaded by the
experiments through the `medmnist` package when required; they remain local and are ignored by
Git.

## Notes on reproducibility

- Training generative models is compute-intensive. A CUDA-capable GPU is recommended.
- Each notebook is a separate experimental path, with its own training settings and evaluation
  flow.
- Historical results from the original academic report are not bundled here and should not be
  treated as newly reproduced benchmark results.
- Notebook outputs and execution history were removed before publication. Run the notebooks to
  generate new results in your own environment.

## Privacy and sharing

No dataset copies, trained weights, generated medical images, personal contact details or secrets
are included. The repository is initially kept private while its contents are reviewed for public
release.
