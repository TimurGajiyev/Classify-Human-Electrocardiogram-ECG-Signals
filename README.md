# ECG Classification Project

## Overview

- Load ECG signal data and preprocess it. This code was created based on an original MATLAB implementation.

- Generate a Short Time Fourier Analysis (Spectrogram) and a Wavelet Analysis (Scalogram).

- Create a dataset using PhysioNet ECG data and convert ECG signals into RGB images for training.

- Design and train CNN models such as GoogLeNet, SqueezeNet, and AlexNet for ECG signal classification.

- Evaluate the models using validation data and compare their performance.

- Classify a given ECG signal using the trained models.

## Usage

1. Ensure you have the necessary Python libraries installed, including NumPy, SciPy, Matplotlib, scikit-learn, Keras, and TensorFlow.

2. Prepare your ECG signal data and PhysioNet dataset (ECGData.mat).

3. Modify the code as needed for your specific dataset and requirements, especially for CNN design and training options.

4. Run the Python script to perform the tasks mentioned in the Overview section.

5. Review the results, including the accuracy of the trained models and their performance in classifying ECG signals.

## Directory Structure

- `data/`: Store your ECG signal data and PhysioNet dataset here.

- `ECGTypesWavelet/` and `ECGTypesSTFT/`: Directories for storing generated RGB images of ECG signals.

- `models/`: Store trained model files here (e.g., GoogLeNetModel2.h5, SqueezeNetModel2.h5, AlexNetModel2.h5).

## Notes

- This Python code was created based on an original MATLAB code for ECG signal classification.

- The code is written in Python and may require adaptations for specific datasets and preferences.

- Ensure you have the required image data in 'ECGTypesWavelet' or 'ECGTypesSTFT' for model training.

- Update the model architecture, hyperparameters, and training options according to your dataset and needs.

## Prerequisites

Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- Required Python libraries (NumPy, SciPy, Matplotlib, scikit-learn, Keras)
- MATLAB (for some parts of the code that were originally written in MATLAB)

## Getting Started

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/TimurGajiyev/Classify_Human_Electrocardiogram_ECG_Signals/
    ```

2. Install the necessary Python libraries:

    ```bash
    pip install numpy scipy matplotlib scikit-learn keras tensorflow
    ```

3. Download the required dataset:
   
    You should have an ECG dataset in MATLAB format (`ecg.mat` and `ECGData.mat`) for the project.

4. Run the Python code:

    - The Python code (`ECG_Project.py`) contains the signal processing, data preprocessing, deep learning model training, and evaluation steps.

    - The code is well-commented to help you understand each section.

## Project Structure

- `ecg_classification.py`: The main Python script for ECG signal classification.
- `ECGData.mat`: The MATLAB dataset containing ECG data.
- `ecg.mat`: Sample ECG signal in MATLAB format.
- `ECGTypesWavelet`: Directory for storing scalogram images.
- `ECGTypesSTFT`: Directory for storing spectrogram images.

## Results

After running the code, you will get accuracy metrics and confusion matrices for the trained deep learning models.

## Acknowledgment

Feel free to customize this README with additional information, credits, and usage instructions as needed for your specific project.
