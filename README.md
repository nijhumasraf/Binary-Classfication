# 🧠 Binary Classification with Multiple ML Models

This project applies **binary classification** techniques on four real-world datasets using various machine learning models, including:

- Decision Tree
- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine (via GridSearchCV)
- K-Fold Cross Validation

## 📁 Datasets Used

1. **Breast Cancer**
2. **Loan Approval**
3. **Stroke Prediction**
4. **Income Classification**

All datasets are expected to be in `.csv` format and stored in the root folder.

## 📦 Libraries Used

- `pandas`, `numpy`
- `matplotlib`
- `scikit-learn`
  - `DecisionTreeClassifier`, `RandomForestClassifier`
  - `LogisticRegression`, `KNeighborsClassifier`, `SVC`
  - `train_test_split`, `cross_val_score`, `KFold`, `GridSearchCV`
  - `accuracy_score`, `classification_report`, `confusion_matrix`
- `IPython.display` for formatted output in notebooks

## ⚙️ Preprocessing Highlights

- Categorical variables are encoded using `category` or mapping.
- Missing values (e.g., BMI in stroke dataset) are filled using mean imputation.
- Only numerical columns are selected for model training.

## 🧪 Models Trained

Each model is evaluated using:
- **Accuracy**
- **Confusion Matrix**
- **Classification Report (Precision, Recall, F1-Score)**
- **Feature Importance** (where supported)
- **Visualization of Feature Importance**

## 🧠 K-Fold Cross Validation

- Logistic Regression is validated using `KFold` (default k=5).
- Gives an average accuracy score to assess model generalizability.

## 🔍 Grid Search for SVM

- SVM model is tuned using `GridSearchCV` with:
  - `C`: [0.25, 0.5, 0.75, 1]
  - `kernel`: ['linear', 'rbf']
- Subsampled (30%) to improve speed.

## 🧾 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/nijhumasraf/Binary-Classfication.git
   cd Binary-Classfication
