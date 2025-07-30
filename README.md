# RDTF

A project for generating text-based dynamic images (Under updating).

## Overview

RDTF (Text-Driven Dynamic Image Generation) is a cutting-edge project focused on generating dynamic images from text descriptions. By leveraging advanced AI models and fine-tuning techniques, RDTF enables the creation of visually engaging dynamic visuals that respond to textual inputs.

The project utilizes the **i2vgen model** as its foundation and incorporates **LoRA (Low-Rank Adaptation)** fine-tuning technology to optimize performance specifically for dynamic image generation tasks. This combination allows for efficient adaptation of the base model while maintaining high-quality output.

## Features

- **Text-to-Dynamic-Image**: Convert text descriptions into dynamic images with smooth animations
- **LoRA Fine-tuning**: Efficient model adaptation using LoRA technology
- **Customizable Training**: Flexible training pipeline for different dynamic image generation tasks
- **Easy Integration**: Simple invocation process for generating dynamic images

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- diffusers library
- Other dependencies (see `requirements.txt`)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/RDTF.git
cd RDTF

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Training

RDTF uses LoRA fine-tuning for optimal performance.

### Training Command

```bash
# Run the training script
bash shells/train_multitaskpretrain.sh
```

### Training Configuration

You can modify the training parameters in the `train_multitaskpretrain.sh` script, including:
- Learning rate
- Number of epochs
- Batch size
- Dataset paths
- LoRA rank and alpha values

## Inference

### Generate Dynamic Images

To generate dynamic images using the trained model:

```bash
/usr/local/envs/diffusers/bin/python examples_lora.py
```

### Custom Inputs

Modify the `examples_lora.py` file to provide your own text prompts and adjust generation parameters such as:
- Output resolution
- Animation length
- Style parameters
- Sampling steps

## Examples

Check out the `examples/` directory for sample outputs and corresponding input prompts.

## Contributing

We welcome contributions to RDTF! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and submission process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Thanks to the developers of the i2vgen model
- LoRA implementation based on [peft](https://github.com/huggingface/peft) library
- Diffusers library by Hugging Face

## Contact

For questions and feedback, please open an issue and contact.
