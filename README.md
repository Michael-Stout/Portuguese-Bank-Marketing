Updated Report Reflecting Final Findings

# Practical Assignment 3 (Module 17)

## Overview

This project focuses on developing and assessing predictive models to enhance customer subscription rates for long-term deposit products in a marketing campaign. The process includes data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

## Data Description

The dataset contains both numerical and categorical features that reflect important aspects of customer behavior.

### Numerical Features
	•	age: Age of the customer
	•	duration: Last contact duration in seconds
	•	campaign: Number of contacts performed during this campaign
	•	pdays: Number of days since the client was last contacted
	•	previous: Number of contacts performed before this campaign
	•	emp.var.rate: Employment variation rate
	•	cons.price.idx: Consumer price index
	•	cons.conf.idx: Consumer confidence index
	•	euribor3m: Euribor 3-month rate
	•	nr.employed: Number of employees

### Categorical Features
	•	job: Type of job
	•	marital: Marital status
	•	education: Education level
	•	default: Has credit in default?
	•	housing: Has housing loan?
	•	loan: Has personal loan?
	•	contact: Type of communication contact
	•	month: Last contact month
	•	day_of_week: Last contact day of the week
	•	poutcome: Outcome of the previous marketing campaign

## Exploratory Data Analysis (EDA)

### Visualizations

The project includes histograms for each numeric column and count plots for categorical variables, revealing a challenging classification problem with imbalanced classes, typical short call durations, and strong relationships between economic indicators. Key observations:
	1.	Class Distribution
	•	Highly imbalanced: ~88.73% “no,” ~11.27% “yes” to term deposits.
	2.	Feature Correlations
	•	Strong economic indicator cluster (e.g., emp.var.rate, euribor3m, nr.employed).
	•	Personal variables like age or duration show minimal direct correlation to each other but still matter for predicting “yes” vs. “no.”
	3.	Demographic Patterns
	•	Age: Large concentration in the 30–60 range.
	•	Education: University degree holders dominate, followed by high school.
	4.	Call Duration
	•	Heavily skewed, with a substantial number of short calls (under ~300 seconds), but a long tail that extends to 5000 seconds.

## Model Comparison and Observing Overfitting

Four models were evaluated: Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM). After hyperparameter tuning, the following table summarizes their final performance:


| Model               | Train Time (s)| Train Accuracy | Test Accuracy | ROC AUC  | Best Params                                                        |
|---------------------|---------------|----------------|---------------|----------|--------------------------------------------------------------------|
| Logistic Regression | 1.81          | 0.880539       | 0.881471      | 0.94377  | {'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}                  |
| Decision Tree       | 0.53          | 1              | 0.972071      | 0.972071 | {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2} |
| KNN                 | 0.20          | 1              | 0.943460      | 0.968451 | {'n_neighbors': 3, 'weights': 'distance'}                          |
| SVM                 | 18.54         | 1              | 0.998638      | 1        | {'C': 10, 'gamma': 1, 'kernel': 'rbf'}                             |



### Observations from the Table
	1.	SVM
	•	Achieves the highest Test Accuracy (≈0.9986) and a perfect ROC AUC of 1.0.
	•	Trains significantly longer (18.54 s) compared to the others.
	•	Best hyperparameters: C=10, gamma=1, kernel='rbf'.
	2.	Decision Tree
	•	Perfect Train Accuracy (1.0), indicating it likely overfits somewhat.
	•	Test accuracy is still strong (≈0.9721), and ROC AUC is still strong (≈0.9721).
	•	Very fast training time (0.53 s).
	3.	KNN
	•	Also hits 1.0 on train accuracy (another sign of potential overfitting).
	•	Has a lower test accuracy (≈0.9435) than the Decision Tree or SVM.
	•	ROC AUC ~0.9685 is still relatively high, and training time is minimal (0.20 s).
	4.	Logistic Regression
	•	Good balance between performance and speed; ~0.88 test accuracy, 0.9438 ROC AUC, finishes in 1.81 s.
	•	Less prone to overfitting than the Tree-based and KNN models since train vs. test accuracy is similar.

### The Conclusion from the Comparison
	•	Support Vector Machine (SVM) yields near-perfect performance on this dataset, making it the top choice.
	•	Decision Tree and KNN demonstrate potential overfitting, shown by perfect training scores but lower test scores.
	•	Logistic Regression remains a strong, interpretable baseline with moderate accuracy and AUC, especially if one values training speed and model simplicity.

## Classification Performance Highlights
	•	SVM stands out with a very high test accuracy (~99.86%) and an ROC AUC of 1.0.
	•	Decision Tree and KNN also show high accuracies but likely overfit.
	•	Logistic Regression is stable, with a solid ~88% test accuracy and fewer discrepancies between training and test scores.

## Feature Importance Analysis

Because SVM performed best, permutation feature importance was generated to understand which features most affect the model’s output, indicating economic factors (emp.var.rate, nr.employed, euribor3m) and call-related features (e.g., duration) are important drivers of term-deposit subscription predictions. Logistic Regression showed similar top features but with less extreme weighting than the SVM-based approach.

## Partial Dependence Plots (PDPs)

The PDPs for key numeric features (e.g., duration, pdays, campaign, emp.var.rate) confirm that:
	1.	Moderate contact duration (around 6–8 minutes) correlates with higher subscription likelihood.
	2.	Infrequent contacts (low campaign, more days between calls) are beneficial.
	3.	Economic Indicators such as emp.var.rate and euribor3m show nuanced effects.

## Business Implications
	1.	Optimize Call Durations: Focus on calls around 6-8 minutes to balance engagement and time investment.
	2.	Reduce Contact Frequency: Multiple contacts can annoy customers; fewer, well-placed calls can lead to better success.
	3.	Monitor Economic Indicators: Consider focusing campaigns when indicators are in a favorable range.
	4.	Timing: Pacing out calls (higher pdays) improves outcomes.

## Alignment with Other Research
	•	The findings are consistent with previous studies showing SVM’s effectiveness for complex, imbalanced data.
	•	Feature significance also aligns with typical bank marketing insights, highlighting call duration and economic conditions as crucial predictors.

## Recommendations
	1.	Use SVM for the highest accuracy and broad coverage of actual subscribers.
	2.	Refine Contact Strategy by adjusting call length and spacing between calls.
	3.	Continue Feature Engineering to capture deeper economic signals and time-based insights (e.g., monthly or quarterly trends).
	4.	Deploy Model in live campaigns with careful monitoring of performance and resource usage (SVM training time is longer but can be offset by more efficient marketing efforts).

The SVM model is recommended for its superior performance and actionable insights. Future campaigns and analyses can benefit from further enhancements while maintaining the powerful classification the current best model provides.
