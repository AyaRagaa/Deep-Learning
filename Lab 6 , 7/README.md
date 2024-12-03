# Image Captioning with Deep Learning

## Project Overview

This project implements an advanced image captioning system using deep learning techniques, specifically leveraging InceptionV3 for feature extraction and a sequential LSTM-based model for caption generation.

## Features

- Image feature extraction using pre-trained InceptionV3 model
- Caption generation using LSTM neural network
- Two caption generation strategies: Greedy Search and Beam Search
- BLEU score evaluation of generated captions
- Live camera caption generation

## Prerequisites

### Libraries
- NumPy
- Pandas
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- NLTK
- OpenCV

### Dataset
- Flickr8k dataset (images and captions)

## Project Structure

### Data Preprocessing
- Image feature extraction
- Caption cleaning and tokenization
- Vocabulary building
- Sequence generation

### Model Architecture
- Encoder: InceptionV3 for image feature extraction
- Decoder: LSTM-based sequential model
- Embedding layer
- Dense layers for caption generation

### Caption Generation Methods
1. **Greedy Search**
   - Generates captions by selecting the most probable word at each step
2. **Beam Search**
   - Explores multiple caption paths to find the most likely sequence

### Evaluation Metrics
- BLEU-1 and BLEU-2 scores for caption quality assessment

## Installation

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install numpy pandas tensorflow matplotlib seaborn nltk opencv-python
   ```
3. Download pre-trained InceptionV3 weights
4. Prepare Flickr8k dataset

## Usage

### Training the Model
- Run the Jupyter notebook to train the image captioning model
- Adjust hyperparameters as needed

### Live Caption Generation
- Use the `live_caption_generator()` function to generate captions from webcam input
- Press 'Space' to capture and generate a caption
- Press 'Q' to quit

## Model Performance Visualization

The project includes visualization of:
- Training and validation loss
- Sample image caption generation
- BLEU score comparisons

## Customization

- Modify model architecture
- Experiment with different feature extraction models
- Adjust beam search parameters

## Limitations

- Depends on pre-trained weights and dataset quality
- Caption generation can be inconsistent
- Computational resources required for training

## Future Improvements

- Implement attention mechanisms
- Use transformer-based architectures
- Incorporate more sophisticated language models

## References

- Flickr8k Dataset
- InceptionV3 Paper
- LSTM and Seq2Seq Learning
