# ECG Classification Project

## Overview

This project focuses on classifying Electrocardiogram (ECG) signals using various signal processing techniques and deep learning models. The goal is to identify the underlying heart condition (e.g., arrhythmia, congestive heart failure, normal sinus rhythm) from the ECG data.

## Prerequisites

Before running the project, make sure you have the following dependencies installed:

- Python 3.x
- Required Python libraries (NumPy, SciPy, Matplotlib, scikit-learn, Keras)
- MATLAB (for some parts of the code that were originally written in MATLAB)

## Getting Started

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/ecg-classification-project.git
    ```

2. Install the necessary Python libraries:

    ```bash
    pip install numpy scipy matplotlib scikit-learn keras tensorflow
    ```

3. Download the required dataset:
   
    You should have an ECG dataset in MATLAB format (`ecg.mat` and `ECGData.mat`) for the project.

4. Run the Python code:

    - The Python code (`ecg_classification.py`) contains the signal processing, data preprocessing, deep learning model training, and evaluation steps.

    - The code is well-commented to help you understand each section.

## Project Structure

- `ecg_classification.py`: The main Python script for ECG signal classification.
- `ECGData.mat`: The MATLAB dataset containing ECG data.
- `ecg.mat`: Sample ECG signal in MATLAB format.
- `ECGTypesWavelet`: Directory for storing scalogram images.
- `ECGTypesSTFT`: Directory for storing spectrogram images.

## Results

After running the code, you will get accuracy metrics and confusion matrices for the trained deep learning models.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://openai.com) for assistance with this project.

Feel free to customize this README with additional information, credits, and usage instructions as needed for your specific project.

