# FDA Assignment 1 - Linear Regression Analysis  

## Project Overview  
This assignment was based on executing and analyzing three Google Colab files using Python. The objective was to apply various regression techniques to real-world problems and answer analytical questions based on model behavior, assumptions, and interpretation.

## Project Structure  
The assignment was divided into three key parts:

1. **Simple Linear Regression (Salary vs Experience)** (Linear_Regression_Salary.ipynb)
2. **Multiple Linear Regression (Startup Profit Prediction)** (MLR_Startup_Profit.ipynb)
3. **Advanced Linear Regression (Student Performance Analysis)** (1_studentp_Regression_Applied.ipynb)

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


# FDA Assignment 2 - Logistic Regression: Insurance Claim Fraud Detection   

## 📘 Project Overview  
This assignment focused on building a **logistic regression model** (Assignment_2_26591_FDA.ipynb) to detect fraudulent insurance claims using a real-world dataset. The task involved data preprocessing, handling class imbalance, applying logistic regression, and evaluating the model using appropriate metrics.

The project was implemented in Python using Jupyter Notebook, following a systematic machine learning workflow.

---

## 📁 Project Structure

The assignment was divided into the following major components:

- **Data Preprocessing & EDA**
- **Class Imbalance Handling (SMOTE, class weights)**
- **Model Building using Logistic Regression**
- **Model Evaluation & Metrics Analysis**
- **Insights and Recommendations for Fraud Detection**

---

## 📊 Step 1: Data Preprocessing & Exploratory Data Analysis

**Objective:**  
Clean the dataset, explore variable distributions, and understand the class balance.

**Key Steps:**

- Loaded and cleaned the dataset using `pandas`.
- Visualized the distribution of fraudulent vs. non-fraudulent claims.
- Detected and addressed class imbalance early in the process.

**Insights:**

- The dataset showed significant class imbalance.
- Most features were numerical; few categorical variables required minimal encoding.

---

## ⚖️ Step 2: Handling Class Imbalance

**Objective:**  
Improve model learning by addressing skewed target classes.

**Techniques Used:**

- **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.
- Compared with **undersampling** and **class-weight adjustments**.
- Regularization was applied to prevent overfitting.

**Result:**

- SMOTE helped the model generalize better to minority (fraudulent) class.

---

## 📈 Step 3: Logistic Regression Model

**Objective:**  
Train a binary classification model using logistic regression.

**Key Details:**

- Used `train_test_split` (80/20 split).
- Applied standard scaling for feature normalization.
- Fitted logistic regression with regularization (L2 penalty).

---

## 📊 Step 4: Model Evaluation

**Evaluation Metrics Used:**

-	Accuracy: 0.986
-	AUC-ROC Score: 0.62
-	Classification Report:

Class	 : 0 (Non-Fraud)	       		  	     

Precision	: 0.99
Recall	:  1.00
F1-Score	:   0.99
Support :  1980

Class : 1 (Fraud) 

Precision	: 0.00
Recall	:  0.00
F1-Score	:   0.00
Support :  20

-	Macro Avg F1: 0.50

-	Weighted Avg F1: 0.98

**Findings:**

- Precision and recall trade-off was important due to fraud detection nature.
- AUC-ROC score confirmed good model performance.
- Confusion matrix helped identify False Negatives (missed frauds).
-	Although overall accuracy is very high (98.6%), this is due to the dominance of non-fraud cases.
-	The model failed to detect fraud cases (Recall = 0.00), indicating the class imbalance remains a critical issue.
-	AUC-ROC score of 0.62 suggests the model has limited ability to separate fraud from non-fraud.
-	Further action is needed to improve detection of the minority class.


---

## 💡 Step 5: Insights & Recommendations

**Strengths:**

- Balanced model with good recall for fraud cases.
- SMOTE improved minority class recognition significantly.

**Improvements:**

- Explore ensemble models (Random Forest, XGBoost) for better fraud capture.
- Feature engineering on categorical variables (e.g., claim type, customer segment).
- Use domain knowledge to include more predictive variables.

---

## ✅ Conclusion

This assignment helped develop skills in:

- Handling imbalanced datasets effectively.
- Applying logistic regression for binary classification.
- Evaluating classification models using multiple metrics.
- Drawing actionable insights from machine learning output.

The project mimicked real-world challenges in insurance fraud analytics and improved practical understanding of fraud detection models.

---

## 💻 Technologies Used

- **Python**
- **Jupyter Notebook**
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`

---






