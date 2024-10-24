# ML_Aufgabe

## Project Overview

This project is a ResNet-based image classification model implemented using the PyTorch framework and the `torchvision` library. The aim is to classify images into multiple categories such as Bicycle, Bus, Traffic Light, etc. The project includes a complete workflow for model training, validation, testing, and generating results, which are finally output into a CSV file.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- OpenCV
- PIL (Pillow)
- Google Colab (optional)

### Installation of Dependencies

You can install the required dependencies using the following command:

```bash
pip install torch torchvision numpy opencv-python Pillow
```

## Project Structure

- `ML_utimi_uktno.ipynb`: The Jupyter notebook containing the core project.
- `train_val_data`: Folder containing the training and validation datasets, with subfolders representing different categories such as Bicycle, Bus, etc.
- `test_data`: Folder containing the test dataset for model evaluation.

## Usage Guide

### 1. Data Preparation

The training and validation datasets, along with the test dataset, can be downloaded from the following link:

[Download Dataset](https://drive.google.com/drive/folders/16VB0OQYX7Pr2VvcfQolr3EmDg7R3wtCJ)

After downloading, upload the zip files directly to your Google Drive (`/content/drive`). Do **not** extract them locally. The project will handle the extraction during runtime.

### 2. Model Training

The `ResNetDataset` class is used to load training and validation data. The training loop uses PyTorch's standard approach, which includes:

- Forward pass
- Loss computation (CrossEntropy Loss)
- Backward pass
- Parameter updates using the Adam optimizer

After each epoch, the model is evaluated on the validation set, and the best model (based on validation loss and accuracy) is saved.

### 3. Model Validation and Inference

After training, the model is used for inference on the test dataset. The classification predictions are saved in a `result1.csv` file in the following format:

```
ImageName,Bicycle,Bus,Traffic_Light,...
test_image_1.png,0.1,0.2,0.7,...
```

Each row represents the predicted probability distribution for a test image.

### 4. Running the Code

If running in Google Colab:

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Unzip the training and test data inside Colab:
   ```bash
   !unzip /content/drive/MyDrive/train_val.zip
   !unzip /content/drive/MyDrive/test.zip
   ```

3. Run the training and validation loop:
   Execute the entire notebook to train the model and save the best model as `best_loss.pt` and `best_accuracy.pt`.

## Results

The model's performance is evaluated during training, and the model with the best validation loss and accuracy is saved. The final output predictions are saved in a CSV file for further analysis.

## Contributors

- This project was developed and reproduced by Yushu Gao.
