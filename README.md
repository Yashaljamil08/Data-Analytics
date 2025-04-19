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
