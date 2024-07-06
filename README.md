# Palestinian Accent Recognition System

## Project Overview
This project aims to develop a system that recognizes different Palestinian accents from Jerusalem, Nablus, Hebron, and Ramallah. The system uses Mel Frequency Cepstral Coefficients (MFCCs) for feature extraction and employs three machine learning classifiers: Support Vector Machine (SVM), Random Forest, and Logistic Regression. The goal is to improve the accuracy of accent recognition and contribute to the research on Palestinian accents and Arabic speech recognition.

## Methodology
### Feature Extraction
- `MFCCs`: Capture the short-term power spectrum of the sound.
- `Spectral Contrast`: Considers the amplitude difference between peaks and valleys in the sound spectrum.

### Classifiers
1. Random Forest Classifier
  - 100 trees
  - Random state: 42
  - Standard Scaler for feature normalization

2. Logistic Regression Classifier
  - Uses probabilities for class prediction
  - Standard Scaler for feature normalization

3. SVM Classifier
  - RBF kernel
  - Regularization parameter (C): 10.0
  - Gamma: 'scale'

## Results and Comparison
| Classifier              | Overall Accuracy | Precision (Hebron) | Recall (Hebron) | F1-Score (Hebron) | Precision (Jerusalem) | Recall (Jerusalem) | F1-Score (Jerusalem) | Precision (Nablus) | Recall (Nablus) | F1-Score (Nablus) | Precision (Ramallah) | Recall (Ramallah) | F1-Score (Ramallah) |
|-------------------------|------------------|-------------------|-----------------|-------------------|-----------------------|---------------------|-----------------------|--------------------|------------------|--------------------|----------------------|--------------------|---------------------|
| Random Forest           | 70.00%           | 71%               | 100%            | 83%               | 60%                   | 60%                 | 60%                   | 100%               | 20%              | 33%                | 71%                  | 100%               | 83%                 |
| Logistic Regression     | 65.00%           | 83%               | 100%            | 91%               | 60%                   | 60%                 | 60%                   | 100%               | 40%              | 57%                | 43%                  | 60%                | 50%                 |
| SVM                     | 75.00%           | 83%               | 100%            | 91%               | 100%                  | 100%                | 100%                  | 100%               | 40%              | 57%                | 43%                  | 60%                | 50%                 |

### Confusion Matrix for SVM Classifier

## Conclusion
The project successfully developed a system to recognize Palestinian accents using three different classifiers. The SVM classifier performed the best in terms of overall accuracy, followed by Random Forest and Logistic Regression. The confusion matrix for the SVM classifier demonstrates its effectiveness in correctly identifying the different accents.

## Contact
For any questions or suggestions, please contact: 
- Mohammed Owda: 1200089@birzeit.edu 
- Mohammad Abu Shams: 1200549@birzeit.edu 
- Mohammad Sabobeh: 1200388@birzeit.edu 
