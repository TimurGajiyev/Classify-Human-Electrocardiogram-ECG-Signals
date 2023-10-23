import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.signal import spectrogram
from scipy.signal import cwt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
import matplotlib.image as mpimg
from keras.preprocessing.image import ImageDataGenerator


# Load ECG signal
ecg_data = scipy.io.loadmat('ecg.mat')
ecg_signal = ecg_data['ecg']

# Resample the ECG signal to match PhysioNet database
ecg_signal = resample(ecg_signal, int(len(ecg_signal) * 2 / 125))
fs = 128

# Graphical Output with Time Signal
t = np.arange(0, len(ecg_signal)) / fs
plt.plot(t[:500], ecg_signal[:500])
plt.title('ECG to be analyzed')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Short Time Fourier Analysis -- Spectrogram
f, t, Sxx = spectrogram(ecg_signal, fs, nperseg=256, noverlap=128)
plt.pcolormesh(t, f, 10 * np.log10(Sxx))
plt.title('Short Time Fourier Analysis (Spectrogram)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# Wavelet Analysis -- Scalogram
scales = np.arange(1, 128)
cwt_result = cwt(ecg_signal, scales, 'morl')
plt.figure(figsize=(6, 4))
plt.imshow(np.abs(cwt_result), extent=[0, len(ecg_signal) / fs, 1, 128], aspect='auto', cmap='jet')
plt.title('Wavelet Analysis (Scalogram)')
plt.xlabel('Time (s)')
plt.ylabel('Scale')
plt.show()




# Data Preprocessing & Datastore
# (Assuming the data generation steps were already completed in MATLAB)

# Directory paths for scalogram and spectrogram images
wavelet_directory = "ECGTypesWavelet"
stft_directory = "ECGTypesSTFT"

# Create directories if they don't exist
os.makedirs(wavelet_directory, exist_ok=True)
os.makedirs(stft_directory, exist_ok=True)

# Load the PhysioNet data
# (Assuming you have 'ECGData.mat' in the working directory)

# Load the PhysioNet data
physionet_data = scipy.io.loadmat('ECGData.mat')

# Extract relevant data
ecg_raw_data = physionet_data['ECGData']['Data'][0][0]  # ECG timeseries
ecg_labels = physionet_data['ECGData']['Labels'][0][0]  # ECG labels

# Sampling frequency
fs = 128

# Clear the workspace variable
del physionet_data

# Data Preprocessing & Visualization
# (You can add code to preprocess and visualize the loaded data)

# Split ECG data into training and validation sets
# (You can use a different method to split the data, e.g., random sampling)
training_ecg_data = ecg_raw_data[:800]  # First 800 samples
validation_ecg_data = ecg_raw_data[800:]  # Remaining samples

# Visualize the ECG data


plt.figure()
plt.plot(training_ecg_data)
plt.title('Sample ECG Data')
plt.xlabel('Time (samples)')
plt.ylabel('Amplitude')
plt.show()

# You can add more preprocessing and data visualization steps here

# Create an image datastore for scalogram and spectrogram images


# Set parameters for data augmentation (if needed)
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Other data preprocessing steps for images can be added as needed

# Apply the data augmentation to the ECG data
# (You may need to reshape the data to match the expected image dimensions)

# Store the generated images in a directory (e.g., 'wavelet_directory' or 'stft_directory')
# Ensure the images are saved with appropriate labels for classification



ecg_data = scipy.io.loadmat('ECGData.mat')
ECGRawData = ecg_data['ECGData']['Data'][0][0]
ECGLabels = ecg_data['ECGData']['Labels'][0][0]
fs = 128

# Plot different available ECGs from PhysioNet to be used in the training phase
unique_ecg_types = np.unique(ECGLabels)

for ecg_type in unique_ecg_types:
    index = np.where(ECGLabels == ecg_type)[0][0]
    plt.figure()
    plt.plot(ECGRawData[index, :500])
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('Amplitude')
    plt.title(f'ECG Type: {ecg_type[0]}')
    plt.show()

# Evaluate one of the training ECG signals
# Assuming you have the training data ready
FilterBank = cwtfilterbank(signal_length=1000, sampling_frequency=fs, voices_per_octave=12)
Signal = ECGRawData[0, :1000]

cfs, frequencies = FilterBank.wt(Signal)
t = np.arange(1000) / fs

plt.figure()
plt.pcolormesh(t, frequencies, np.abs(cfs))
plt.gca().set_yscale('log')
plt.gca().set_aspect('auto')
plt.title('Scalogram (Example for one PhysioNet ECG)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

# Generate RGB images (scalograms) of all the ECG signals in the PhysioNet database
image_directory = "ECGTypesWavelet"

if not os.path.exists(image_directory):
    os.makedirs(image_directory)

for i in range(len(ECGRawData)):
    cfs = np.abs(FilterBank.wt(ECGRawData[i, :]))
    im = plt.get_cmap('jet')(cfs / np.max(cfs))
    mpimg.imsave(f"{image_directory}/{ECGLabels[i][0]}_{i}.jpg", im)

# STFM
L = 32
dN = 0.75 * L
N = 2 * L
Window = np.hamming(L)
Overlap = int(L - dN)

STFTDirectory = "ECGTypeSTFT"

if not os.path.exists(STFTDirectory):
    os.makedirs(STFTDirectory)

for i in range(len(ECGRawData)):
    f, t, Sxx = spectrogram(ECGRawData[i], fs, window=Window, nperseg=L, noverlap=Overlap, nfft=N)
    spectrum = np.abs(Sxx)
    im = plt.get_cmap('jet')(spectrum / np.max(spectrum))
    mpimg.imsave(f"{STFTDirectory}/{ECGLabels[i][0]}_{i}.jpg", im)

# Generate datastore for training the model
# Assuming you have image files in 'ECGTypesWavelet' or 'ECGTypesSTFT'

image_directory = "ECGTypesWavelet"
image_size = (224, 224)

if not os.path.exists(image_directory):
    print(f"Image directory '{image_directory}' not found.")
else:
    from keras.preprocessing.image import ImageDataGenerator

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(image_directory, target_size=image_size, batch_size=32, class_mode='categorical', subset='training')
    val_generator = datagen.flow_from_directory(image_directory, target_size=image_size, batch_size=32, class_mode='categorical', subset='validation')

# Define and train the InceptionV3 model (you can adjust the architecture and hyperparameters)
num_classes = len(train_generator.class_indices)
base_model = InceptionV3(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Test and validate the model using the validation dataset
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation accuracy: {val_acc}")

# Apply the model to classify a given ECG signal (you need to have the signal and convert it into an image)
# Load and preprocess the ECG signal image to match the model's input requirements
ecg_image_path = 'ECG_To_EvaluateSTFT.jpg'  # Change to the path of the generated image
img = image.load_img(ecg_image_path, target_size=image_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.  # Normalize the image

# Use the trained model to predict the class of the ECG signal image
class_predictions = model.predict(img_array)

# Assuming you have a list of class labels corresponding to your model's output
class_labels = list(train_generator.class_indices.keys())

# Get the predicted class label
predicted_class = class_labels[np.argmax(class_predictions)]

print(f'Predicted ECG Class: {predicted_class}')
