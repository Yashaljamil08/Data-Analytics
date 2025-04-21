# FDA Assignment 1 - Linear Regression Analysis  
**Name:** Yashal Jamil  
**ERP:** 26591  

## Project Overview  
This assignment was based on executing and analyzing three Google Colab files using Python. The objective was to apply various regression techniques to real-world problems and answer analytical questions based on model behavior, assumptions, and interpretation.

## Project Structure  
The assignment was divided into three key parts:

1. **Simple Linear Regression (Salary vs Experience)**
2. **Multiple Linear Regression (Startup Profit Prediction)**
3. **Advanced Linear Regression (Student Performance Analysis)**

Each section included model implementation, result interpretation, and responses to relevant conceptual and practical questions.

## Part 1: Linear Regression - Salary Prediction

**Objective:**  
Predict salary based on years of experience using simple linear regression.

**Key Learnings:**  
- A positive coefficient for experience indicates that salary increases with experience.
- Checking distribution is crucial to ensure linearity and avoid skewed results.
- Not having a test set leads to overfitting and biased model evaluation.
- Outliers and bias in prediction suggest model limitations or missing variables.
- More training data helps generalize better, while less data may lead to poor accuracy.

## Part 2: Multiple Linear Regression - Startup Profit Prediction

**Objective:**  
Predict profit based on spending in R&D, Administration, and Marketing.

**Key Insights:**  
- R&D Spend showed the highest impact on profit, while Administration had minimal correlation.
- Marketing Spend had a wide range, indicating potential outliers.
- Categorical variables like "Industry" need to be encoded (e.g., one-hot encoding).
- Using `train_test_split()` helps evaluate model generalization (default split 75/25).
- The model may not transfer well to other countries due to differing economic contexts.

## Part 3: Advanced Linear Regression - Student Performance

**Objective:**  
Predict the student performance index based on multiple study-related factors.

**Variables Used:**  
- Hours Studied  
- Previous Scores  
- Extracurricular Activities  
- Sleep Hours  
- Sample Papers Practiced  

**Findings:**  
- Coefficients show the weight of each predictor on performance.
- P-values indicate statistical significance of predictors.
- More features can improve accuracy, but too many irrelevant ones can cause overfitting.
- High training accuracy but low test accuracy means overfitting.

## Conclusion

This assignment helped apply foundational and advanced regression techniques and evaluate the effects of data quality, variable significance, and model tuning. Each section provided hands-on experience with real-world datasets and enhanced interpretation skills relevant to business and academic scenarios.

## Technologies Used  
- Python  
- Google Colab  
- Libraries: `pandas`, `matplotlib`, `seaborn`, `sklearn`, `statsmodels`




# 🚨 Insurance Claim Fraud Detection – Logistic Regression  
*FDA Assignment 2 | Yashal Jamil – ERP: 26591*

## 📌 Objective

The goal of this assignment was to **build a logistic regression model** that classifies insurance claims as **fraudulent (1)** or **non-fraudulent (0)** using the provided dataset. This involved applying machine learning concepts such as data preprocessing, class balancing, model training, and evaluation.


---

## ⚙️ Workflow Summary

### 1. 🧹 Data Preprocessing & EDA
- Cleaned the dataset and handled missing values.
- Analyzed class distribution of the target variable.
- Visualized insights using matplotlib/seaborn.

### 2. ⚖️ Handling Class Imbalance
- Applied **SMOTE** to address imbalance in fraudulent vs. non-fraudulent cases.
- Explored alternatives like undersampling and class weights.

### 3. 📈 Logistic Regression Model
- Implemented logistic regression using an 80/20 train-test split.
- Optimized the model with regularization.

### 4. 📊 Model Evaluation
- Evaluated using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **AUC-ROC**
- Visualized confusion matrix to analyze false positives/negatives.

### 5. 💡 Insights & Recommendations
- Highlighted model strengths (e.g., fraud detection sensitivity).
- Suggested improvements like feature engineering and trying ensemble models.

---

## 🧠 Key Learnings

- Applied theoretical knowledge from class on logistic regression and model evaluation.
- Understood the challenges of real-world fraud detection, especially class imbalance.
- Gained hands-on experience with `sklearn`, SMOTE, and visualization libraries.


> **Note:** This project was completed as part of the Financial Data Analytics course at IBA. All code, insights, and conclusions are the result of independent work.


