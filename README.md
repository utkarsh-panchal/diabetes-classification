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


---

## Getting Started

### Prerequisites

Ensure you have Python installed. Install the required libraries using:
```bash
pip install -r requirements.txt
