# diabetes-classification
A machine learning project to predict diabetes using a Support Vector Machine (SVM) model. The dataset includes health parameters like glucose, blood pressure, BMI, and age, and the model is trained to classify patients as diabetic or non-diabetic. Includes data preprocessing, model training, evaluation, and a prediction system.

---

## Overview

The goal of this project is to:
1. Preprocess and standardize patient health data.
2. Train a machine learning model using **SVM** to classify patients as **diabetic** or **non-diabetic**.
3. Build a prediction system where users can input new data to get a prediction.

---

## Dataset

The dataset used for this project is the **Diabetes Dataset**, which contains:
- **Rows:** 768 (each row is a patient record)
- **Features:** 8 health parameters
- **Target:** Outcome (0 for Non-diabetic, 1 for Diabetic)

### Sample Features:
| Pregnancies | Glucose | Blood Pressure | Skin Thickness | Insulin | BMI  | Diabetes Pedigree Function | Age | Outcome |
|-------------|---------|----------------|----------------|---------|------|----------------------------|-----|---------|
| 6           | 148     | 72             | 35             | 0       | 33.6 | 0.627                      | 50  | 1       |
| 1           | 85      | 66             | 29             | 0       | 26.6 | 0.351                      | 31  | 0       |

---

## File Structure

diabetes_prediction/
│
├── LICENSE                      # License file
├── README.md                    # Project documentation
├── diabetes_prediction.ipynb    # Jupyter Notebook for training and predictions
├── diabetes_prediction_dataset  # Dataset for classification
                 

---

## Getting Started

### Prerequisites

Ensure you have Python installed. Install the required libraries using:
```bash
pip install -r requirements.txt
```

--- 

Running the Project:

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```
2. Open the Jupyter Notebook to explore the project:
   ```
   jupyter notebook diabetes_prediction.ipynb
   ```
3. Run all cells in the notebook to train the model, evaluate its performance, and make predictions.

---

Model Training
The project uses Support Vector Machine (SVM) with a linear kernel to classify patients as diabetic or non-diabetic. Key steps include:

Standardizing features using StandardScaler.
Splitting the dataset into training (80%) and testing (20%) sets.
Training the model using the SVC class from scikit-learn.

Example Code Snippet:
```
from sklearn import svm
classifier = svm.SVC(kernel="linear")
classifier.fit(X_train, y_train)
```

---

Model Evaluation
The model achieved the following accuracy:

Training Accuracy: 78.41% 

Test Accuracy: 76.29%

---

Prediction System
You can input patient data into the trained model to get a prediction. For example:

```
input_data = (6, 190, 92, 0, 0, 35.5, 0.278, 66)
input_array = np.array(input_data).reshape(1, -1)
input_scaled = scalar.transform(input_array)
prediction = classifier.predict(input_scaled)

if prediction[0] == 1:
    print("The patient is diabetic.")
else:
    print("The patient is non-diabetic.")
```

---

Future Work
Potential improvements include:

Using non-linear kernels like RBF for SVM to capture complex relationships.
Implementing cross-validation for better model generalization.
Adding new features or engineering existing ones to improve predictive accuracy.

---

License
This project is licensed under the MIT License - see the LICENSE file for details.

---

Acknowledgments
Dataset: Kaggle Pima Indians Diabetes Database
Libraries: scikit-learn, pandas, numpy, matplotlib



