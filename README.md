# Watermarking Adaptive to the Resolution of Image

---

This code is the implementation of **Convolutional Neural Network-Based Digital Image Watermarking Adaptive
to the Resolution of Image and Watermark** [1].This paper also proposes a neural network to perform a robust,
invisible blind watermarking for digital images

## Content

### Structure

```
watermark_resolution
│   README.md
└───data
│   └───Training
│   └───Testing
└───docs
│   applsci-10-06854.pdf
└───models
└───notebooks
└───src
```

### Usage

To train the network:
```
python src/scripts/train_watermark.py --config src/config/train_watermark.json --name Alpha
```

The hyper-prameters for training is define in `src/config/train_watermark.json`   
The data must be in the `data` directory

## Reference
Jae-Eun Lee, Young-Ho Seo and Dong-Wook Kim;  
"Watermarking Adaptive to the Resolution of Image";  
Department of Electronic Materials Engeering, Kwangwoon University, Seoul 01897, Korea