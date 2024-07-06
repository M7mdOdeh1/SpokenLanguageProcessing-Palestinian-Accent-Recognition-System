import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog, messagebox

from sklearn.decomposition import PCA



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

def load_data(data_path):
    X, y = [], []
    accents = ['Jerusalem', 'Nablus', 'Hebron', 'Ramallah_Reef']
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

# Paths to the dataset
training_path = "training_data"
testing_path = "testing_data"


# Load and preprocess data
X_train, y_train = load_data(training_path)
X_test, y_test = load_data(testing_path)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM Classifier with RBF kernel
svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

dictionary = {'Jerusalem': [], 'Nablus': [], 'Hebron': [], 'Ramallah_Reef': []} 
for i in range(len(y_pred)):
    dictionary[y_test[i]].append(y_pred[i])

print(dictionary)

print("Actual vs Predicted Accents:")
for key in dictionary:
    print(f"{key}: {dictionary[key]}")
    

    

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Dimensionality reduction for visualization
pca = PCA(n_components=2)
X_combined_scaled = np.vstack((X_train_scaled, X_test_scaled))
y_combined = np.hstack((y_train, y_test))
X_combined_pca = pca.fit_transform(X_combined_scaled)

# Extracting indices for training and testing set in the combined array
train_indices = range(len(y_train))
test_indices = range(len(y_train), len(y_train) + len(y_test))

# Plotting
plt.figure(figsize=(10, 7))
colors = {'Hebron': 'green', 'Jerusalem': 'red', 'Nablus': 'blue', 'Ramallah_Reef': 'purple'}
for accent, color in colors.items():
    # Plot training points
    plt.scatter(X_combined_pca[train_indices, 0][y_train == accent], X_combined_pca[train_indices, 1][y_train == accent], color=color, label=f"{accent} (Train)", alpha=0.5)
    # Plot testing points
    plt.scatter(X_combined_pca[test_indices, 0][y_pred == accent], X_combined_pca[test_indices, 1][y_pred == accent], color=color, label=f"{accent} (Test)", marker='^')

plt.title('PCA of Accents (Train and Test)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix and Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=['Hebron', 'Jerusalem', 'Nablus', 'Ramallah_Reef'], yticklabels=['Hebron', 'Jerusalem', 'Nablus', 'Ramallah_Reef'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix with City Labels')
plt.show()


def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
    if file_path:
        features = extract_features(file_path)
        if features is not None:
            features_scaled = scaler.transform([features])
            accent = svm_model.predict(features_scaled)
            messagebox.showinfo("Accent Prediction", f"The predicted accent is: {accent[0]}")
        else:
            messagebox.showerror("Error", "Failed to extract features from the audio file.")



root = tk.Tk()
root.title("Accent Detection")
root.geometry("300x150")

label = tk.Label(root, text="Browse an audio file to detect its accent")
label.pack()

browse_button = tk.Button(root, text="Browse File", command=browse_file)
browse_button.pack(pady=20)

root.mainloop()


