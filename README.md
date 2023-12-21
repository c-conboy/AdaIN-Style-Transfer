# Neural Style Transfer with AdaIN

This repository contains a Python script for performing Neural Style Transfer using the Adaptive Instance Normalization (AdaIN) technique. The script uses a pre-trained encoder and decoder for image style transfer.

## Usage

### Prerequisites

- Python 3
- PyTorch
- Pillow (PIL)

### Installation

```
pip install -r requirements.txt
```

###Running the Script

```
python style_transfer.py -content_image path/to/content_image.jpg -style_image path/to/style_image.jpg -encoder_file path/to/encoder_weights.pth -decoder_file path/to/decoder_weights.pth -alpha 0.8 -cuda y
```

**Arguments**
-content_image: Path to the content image for style transfer.
-style_image: Path to the style image.
-encoder_file: Path to the pre-trained encoder weights file.
-decoder_file: Path to the pre-trained decoder weights file.
-alpha: Level of style transfer, a value between 0 and 1.
-cuda: Use CUDA for faster processing (optional, 'y' or 'n').

###Output
The script will save the stylized image in the ./output/ directory with a filename indicating content, style, and alpha values.

Example

```
python style_transfer.py -content_image content.jpg -style_image style.jpg -encoder_file encoder.pth -decoder_file decoder.pth -alpha 0.7 -cuda y
```

This will save the stylized image as content_style_alpha_0.7.jpg in the ./output/ directory.

Acknowledgments
The script is based on the AdaIN (Adaptive Instance Normalization) network.
