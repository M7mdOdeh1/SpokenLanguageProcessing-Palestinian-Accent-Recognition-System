import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to extract features from an audio file using various features
def extract_features(file_path):
    # Initialize default values to handle exceptions gracefully
    features = None

    # Try to load and process the audio file
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

        # Extract Mel-Frequency Cepstral Coefficients (MFCC)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=100)
        # Compute the mean across time to reduce the feature dimension
        mfccs_mean = np.mean(mfccs, axis=1)

        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(contrast, axis=1)

        # Combine MFCC and spectral contrast features into a single vector
        features = np.hstack((mfccs_mean, contrast_mean))

    except Exception as e:
        # Log any exceptions that occur during the feature extraction
        print(f"Error encountered while parsing file: {file_path}")
        print(f"Exception: {e}")

    return features

# Load dataset and extract features
def load_data(data_path):
    X, y = [], []
    accents = ['Jerusalem', 'Nablus', 'Hebron', 'Ramallah_Reef']  # Update folder names if different
    for accent in accents:
        print(f"Processing {accent}...")
        accent_path = os.path.join(data_path, accent)
        for filename in os.listdir(accent_path):
            print("Processing file: ", filename)
            if filename.endswith('.wav'):
                file_path = os.path.join(accent_path, filename)
                features = extract_features(file_path)
                X.append(features)
                y.append(accent)
    return np.array(X), np.array(y)

# Path to the dataset
training_path = "training_data"
testing_path = "testing_data"

# Load training and testing data
print("Loading training data...")
X_train, y_train = load_data(training_path)
print("Loading testing data...")
X_test, y_test = load_data(testing_path)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test_scaled)

# Calculate the accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))



# GUI for file browsing and accent prediction
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        features = extract_features(file_path)
        if features is not None:
            features_scaled = scaler.transform([features])
            predicted_accent = rf_model.predict(features_scaled)
            messagebox.showinfo("Accent Prediction", f"The predicted accent is: {predicted_accent[0]}")

# Setting up the GUI window
root = tk.Tk()
root.title("Accent Detection")
root.geometry("300x150")

label = tk.Label(root, text="Browse an audio file to detect its accent")
label.pack(pady=20)

browse_button = tk.Button(root, text="Browse File", command=browse_file)
browse_button.pack(pady=20)

root.mainloop()