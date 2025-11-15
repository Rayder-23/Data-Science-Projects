# Data Science & Machine Learning Projects

This repository contains a collection of Python Projects for Data Science, Analysis, Preprocessing, and Machine Learning. It covers essential Python libraries, Data Manipulation techniques, Scaling, Regression, Classification, and Practical Applications on Real Datasets.

---

## Projects Structure:

1. **Basic Libraries**  
   Fundamental Python libraries:  
   - `NumPy` – Numerical Computing  
   - `Matplotlib` – Basic Plotting  
   - `Seaborn` – Advanced Visualizations  

2. **Pandas Project**  
   - Analysis on `car_data.csv` using Pandas.  
   - Demonstrates 30 commonly used Pandas functions for Data Manipulation.  

3. **Scaling**  
   - Explains Standardisation and Normalisation.  
   - Mathematical walkthrough and Python implementation.  

4. **Regression Project**  
   - Applies Linear and Logistic Regression on `lifestyle.csv`.  
   - Focus on model training, prediction, and evaluation metrics.  

5. **Iris Analysis**  
   - Quick analysis of the built-in Iris Dataset.  
   - Logistic Regression (Classification).  

6. **Decision Tree & Random Forest Project**  
   - Applied Decision Tree and Random Forest Classifiers on `camera_dataset.csv`.  
   - Evaluation using Accuracy and Feature Importance.  

7. **KNN Project**  
   - Explains basics of K-Nearest Neighbors (KNN) for Regression and Classification.  
   - Applied on `_bills.csv` Dataset to classify fake or real bills.  

8. **E-Challan Cameras Project**  
   - In-depth Analysis on `cameras.csv` to identify the best cameras for a Traffic E-Challan System.  
   - **Preprocessing & Feature Engineering**  
   - **Rule-based Suitability Score using MinMax Scaling**  
   
   #### ML Applications:
   - **Regression:** Predict the rule-based score using `LinearRegression` and `RandomForestRegressor`.

   - **Classification:** Convert score to classes (`High`, `Medium`, `Low`) and predict using `LogisticRegression` and `RandomForestClassifier`.  
   
   #### Results & Insights:
   - Regression achieved **R² > 0.98** and Classification achieved **97% Accuracy**.  
   - Key features influencing suitability: `Low and Effective Resolution`.  
   - Combines rule-based and ML methods for **transparent, data-driven decisions**.

---

## Requirements

- Python 3.10+  
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

You can install dependencies via:

```bash
pip install -r requirements.txt