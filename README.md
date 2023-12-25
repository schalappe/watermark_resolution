# Watermarking adaptive to the resolution of image

## Table of Contents

- [Introduction](#introduction)
- [Initialize](#initialize)
- [Dataset](#dataset)
- [Hyperparameter Search](#hyperparameter-search)
- [Usage](#usage)
- [Results](#results)
- [Reference](#reference)

---

## Introduction

This code implements the research paper **Convolutional Neural Network-Based Digital Image Watermarking Adaptive
to the Resolution of Image and Watermark**[^1]. This paper proposes a neural network for performing robust, invisible
blind watermarking on digital images. This implementation is build with Keras and Tensorflow as backend.

The project aims to implement the neural network proposed in the research paper and includes a script to search for
optimal hyper-parameters.

Specifically, two neural networks are proposed: one to encode the mark in the image and the other to extract it.

This implementation does not revise the proposed architecture but focuses on finding the hyper-parameters needed to
train an optimal model.

## Initialize

To start the project and install the required packages:

```bash
make initialize
make create_virtualenv
```

## Dataset

The authors used the BOSS dataset[^2] for their entire experiment. However, for this implementation, the following data
will be used:

- For training and hyper-parameters
  search: [Grayscale Image Aesthetic 10K](https://huggingface.co/datasets/ioclab/grayscale_image_aesthetic_10k)
- For testing: [Grayscale Image Aesthetic 6K](https://huggingface.co/datasets/ioclab/grayscale_image_6k)

To download the necessary data:

```bash
make get_data
```

### Structure

```
watermark_resolution
└───data
│   └───train
│   └───tests
└───docs
│   applsci-10-06854.pdf
└───models
    └───params
    └───storage
    └───best_params.json
    └───search_space.json
└───notebooks
└───src
    └───addons
    └───data
    └───scripts
```

## Hyperparameter Search

The process of searching for hyper-parameters involves two steps. Firstly, searching for model-related parameters like
batch, epochs, optimizer, ...:

```bash
make search_model
```

and secondly, searching for parameters related to the loss function:

```bash
make search_loss
```

Optuna[^3] is used for all searches.

## Usage

After finding the hyper-parameters, enrich the JSON file `best_params.json`. Optuna Dashboard can be useful for this:

```bash
make dashboard
```

To train the networks:

```bash
make train_model
```

Models are saved in the `models/storage` directory for future use.

## Results

The results of the project are available in the associated notebooks.

```bash
make notebook
```

## Reference

Lee, Jae-Eun, Young-Ho Seo, and Dong-Wook Kim. 2020. "Convolutional Neural Network-Based Digital Image Watermarking
Adaptive to the Resolution of Image and Watermark" Applied Sciences 10, no. 19:6854. https://doi.org/10.3390/app10196854

[^1]: https://www.mdpi.com/2076-3417/10/19/6854

[^2]: http://decsai.ugr.es/cvg/CG/base.html

[^3]: https://optuna.org/