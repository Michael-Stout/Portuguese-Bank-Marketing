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


---

### Report for Practical Assignment III (Module 17)

**Executive Summary**
---
This project focuses on developing and assessing predictive models to enhance customer subscription rates for long-term deposit products in a marketing campaign. The process includes data preprocessing, exploratory data analysis (EDA), model training, and performance evaluation.

**Data**: 4,119 examples, 21 features, 1,230 missing entries.

**Best Model**: Tuned SVM with 98.77% test accuracy and 0.9973 ROC AUC.
**Decision Tree**: Second place (94.55%), very fast but prone to overfitting.
**KNN**: 90.94% test accuracy, also near-perfect fit on training.
**Logistic Regression**: 74.66% accuracy, stable and interpretable, but lags behind tree/kernel-based models.

**Recommendation**
---
Utilize SVM for best performance, or Decision Tree for speed with minimal performance drop.

While SVM is the champion, it’s also the slowest. Running the code on the larger version of the dataset was cancelled after over 12 hours of processing and would require distributed training or more computational resources to complete.

**Overview**
---
In this analysis, I explore how to predict potential subscriptions to a long-term deposit product using new data (4,119 records, 21 columns) from a Portuguese bank marketing campaign. I chose the smaller dataset as there weren’t significant changes to the training ac**ANALYSIS COMPARISON**

The pipeline involves data cleaning, exploratory data analysis (EDA), various classification models, and hyperparameter tuning.

**Data Description**
	* 	**Shape**: (4119, 21)
	* 	**Null Values**: 1230 missing entries
	* 	**Column Data Types**:
	* 	11 object (categorical)
	* 	5 int64 (integer)
	* 	5 float64 (numeric/floating)

**Key Numerical Features**:
	* 	age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed, etc.

**Key Categorical Features**:
	* 	job, marital, education, default, housing, loan, contact, month, day_of_week, poutcome

**Summary of Notable Distributions**
	* 	**job**: ~24.57% admin., 21.46% blue-collar, 16.78% technician, etc.
	* 	**housing**: ~52.80% yes, ~44.65% no, ~2.55% missing (NaN)
	* 	**loan**: ~81.31% no, ~16.14% yes, ~2.55% missing
	* 	**target (y)**: Split was roughly 11% “yes” vs. 89% “no” before upsampling.
	* 	**Baseline Dummy Classifier**: 0.50 accuracy with upsampled data.

**Exploratory Data Analysis (EDA)**
---
	* 	**Shape & Missingness**: With 4,119 rows and 21 columns, there are 1,230 missing values spread mostly in default, housing, loan, and some education.
	* 	**Numerical Stats**:
	* 	Average age is ~40, ranging 18–88.
	* 	duration (excluded for real modeling) was originally up to ~3,600 seconds.
	* 	euribor3m & emp.var.rate strongly correlate with economic conditions.
	* 	**Categorical Distributions**:
	* 	“admin.” job is largest (24.57%), “blue-collar” next (21.46%).
	* 	“married” is the largest marital category (60.91%), with ~0.27% missing.
	* 	**Target Variable**: For the raw data, ~11% “yes” to term deposit vs. ~89% “no”, but an upsampling approach balanced classes to 50/50 in the training set.

Visualizations
Age

Descriptive Statistics for age:
{'count': 4119.0, 'mean': 40.11361981063365, 'std': 10.313361547199813, 'min': 18.0, '25%': 32.0, '50%': 38.0, '75%': 47.0, 'max': 88.0}
Shape: Right-skewed, with most ages clustering between the late 20s and early 40s.
Implication: The population skews younger to middle-aged, but there are still some older customers in the dataset.

Duration

Descriptive Statistics for duration:
{'count': 4119.0, 'mean': 256.7880553532411, 'std': 254.70373612073678, 'min': 0.0, '25%': 
Shape: Strongly right-skewed, with a large majority of call durations under ~500 seconds and a very long tail of longer calls.
Implication: Most calls are relatively short, but a small number can last much longer.

Campaign

Descriptive Statistics for campaign:
{'count': 4119.0, 'mean': 2.537266326778344, 'std': 2.568159237578138, 'min': 1.0, '25%': 1.0, '50%': 2.0, '75%': 3.0, 'max': 35.0}
Shape: Heavily right-skewed, with most clients contacted fewer than ~5 times, and a small fraction contacted many more times.
Implication: The typical client receives a limited number of calls, but some clients have been contacted very frequently.

Pdays

Descriptive Statistics for pdays:
{'count': 4119.0, 'mean': 960.4221898519058, 'std': 191.92278580077644, 'min': 0.0, '25%': 999.0, '50%': 999.0, '75%': 999.0, 'max': 999.0}
Shape: Nearly all values are around 999, indicating most clients had not been previously contacted or had a large gap since last contact. A small portion has pdays near zero or other small numbers.
Implication: The majority are “new” contacts for this campaign (or far removed from past campaigns).

Previous

Descriptive Statistics for previous:
{'count': 4119.0, 'mean': 0.19033746054867687, 'std': 0.5417883234290308, 'min': 0.0, '25%': 0.0, '50%': 0.0, '75%': 0.0, 'max': 6.0}
Shape: Dominated by 0 (no previous contacts), with a small portion of 1 or 2, and very few above 2.
Implication: Most customers did not have contacts in earlier campaigns.

Emp.var.rate

Descriptive Statistics for emp.var.rate:
{'count': 4119.0, 'mean': 0.08497208060208788, 'std': 1.5631144559116763, 'min': -3.4, '25%': -1.8, '50%': 1.1, '75%': 1.4, 'max': 1.4}
Shape: Shows distinct peaks around −2 and +1.4, indicating certain economic conditions were more common in the data.
Implication: Economic circumstances varied but often clustered around these two states.

Cons.price.idx

Descriptive Statistics for cons.price.idx:
{'count': 4119.0, 'mean': 93.57970429715951, 'std': 0.5793488049889662, 'min': 92.201, '25%': 93.075, '50%': 93.749, '75%': 93.994, 'max': 94.767}
Shape: Multiple peaks between ~92.5 and 94.5, reflecting different “clusters” of consumer price index values over time.
Implication: The CPI shifted in steps, possibly tied to different periods in the dataset.

Cons.conf.idx

Descriptive Statistics for cons.conf.idx:
{'count': 4119.0, 'mean': -40.49910172371935, 'std': 4.594577506837543, 'min': -50.8, '25%': -42.7, '50%': -41.8, '75%': -36.4, 'max': -26.9}
Shape: Multiple peaks roughly between −50 and −35, showing a few dominant confidence levels.
Implication: Consumer confidence was generally negative, fluctuating within a limited range.

Euribor3m

{'count': 4119.0, 'mean': 3.621355668851663, 'std': 1.7335912227013557, 'min': 0.635, '25%': 1.334, '50%': 4.857, '75%': 4.961, 'max': 5.045}
Shape: Two main clusters—one around ~1 and another near ~5—indicating interest rates were either quite low or relatively high, with a “gap” in the middle.
Implication: Distinct macroeconomic environments during the observed periods.

Nr.employed

Descriptive Statistics for nr.employed:
{'count': 4119.0, 'mean': 5166.481694586065, 'std': 73.66790355721253, 'min': 4963.6, '25%': 5099.1, '50%': 5191.0, '75%': 5228.1, 'max': 5228.1}
Shape: Bimodal, with large peaks around ~5100 and ~5200, and smaller peaks under 5000.
Implication: Employment figures changed sharply over time, suggesting different labor-market conditions across the dataset.

**Modeling Approaches**
---

I used **four classifiers**:
	1.	**Logistic Regression**
	2.	**Decision Tree**
	3.	**K-Nearest Neighbors (KNN)**
	4.	**Support Vector Machine (SVM)**

**Imbalanced Data**
Before training, I upsampled the minority class to handle the imbalance. Then I split data (80/20) into train/test sets.

**Evaluation Metrics**
	* 	Accuracy
	* 	ROC AUC
	* 	Precision, Recall, F1 (seen in classification reports)

**Default Results**
---
I obtained the follwing results using the default settings for the models.

**Key Observations**:
	* 	**Decision Tree** is extremely accurate on training (99.97%) and very strong on test (94.55%), indicating some overfitting.
	* 	**KNN** also shows high train accuracy (90.08%) but drops to 85.90% on test.
	* 	**Logistic Regression** is more balanced, ~74.66% test accuracy, with a moderate ROC AUC (~0.7966).
	* 	**SVM** with default hyperparameters underperforms the Decision Tree and KNN in test accuracy (79.63%).

**6. Hyperparameter Tuning Resultes**
The following results were achieved after hyperparameter tuning.

**Findings**:
	* 	**SVM** soared after tuning, reaching **98.77%** test accuracy and **0.9973 ROC AUC**, the best overall.
	* 	**Decision Tree** remains second at **94.55%** test accuracy.
	* 	**KNN** improved to **90.94%** test accuracy.
	* 	**Logistic Regression** stays around **74.66%**, still stable but behind the tree-based or kernel-based methods.

**7. Insights into the New Data and Results**
	1.	**Shape & Missingness**: With 4,119 records, the dataset is moderately sized; 1,230 nulls highlight the importance of careful imputation (especially for default, housing, loan, and some education).
	2.	**Class Distribution**: The raw data is heavily skewed toward “no,” but upsampling balanced the training set.
	3.	**SVM** emerges as the best performer with near-perfect test accuracy (98.77%) and ~1.00 ROC AUC, though it requires **longer training** (over 60 seconds in tuning).
	4.	**Decision Tree** is simpler and very fast to train, scoring around 94.55% on the test set, but it overfits drastically (nearly 100% on training).
	5.	**KNN** also hits perfect training accuracy but dips to ~90.94% test.
	6.	**Logistic Regression** remains the most interpretable, though with modest ~74.66% test accuracy, it might be favored for speed and transparency in certain business contexts.

**Business Recommendations**
---
	* 	**Use SVM** if maximum predictive power is desired and computation time is acceptable.
	* 	**Watch for Overfitting** in the Decision Tree and KNN; while these show strong test performance, they can degrade if future data differs from the training set.
	* 	**Focus on Data Quality**: 1,230 missing entries are significant; refining or collecting missing info could further improve results.
	* 	**Leverage Economic Indicators** (emp.var.rate, euribor3m) and call features (e.g., duration) more explicitly—SVM permutations indicate these are key.
	* 	**Iterate** with advanced feature engineering, model ensembles, or time-series analysis for additional gains.

### New Report Summary
	* 	**New Data**: 4,119 examples, 21 features, 1,230 missing entries.
	* 	**Best Model**: Tuned SVM with 98.77% test accuracy and 0.9973 ROC AUC.
	* 	**Decision Tree**: Second place (94.55%), very fast but prone to overfitting.
	* 	**KNN**: 90.94% test accuracy, also near-perfect fit on training.
	* 	**Logistic Regression**: 74.66% accuracy, stable and interpretable, but lags behind tree/kernel-based models.
	* 	**Recommendation**: SVM for best performance or Decision Tree for speed with minimal performance drop.


### 5. Additional Insight About the New Data
	1.	**Null Values**: With 1,230 missing values out of 4,119 records (~30%), proper imputation or domain-driven treatment is critical to ensure robust models. A large portion of these come from placeholders like “unknown” in columns default, housing, loan, and education.
	2.	**Stronger Economic Shifts**: The emp.var.rate has a wider range here (−3.4 to +1.4), suggesting data covering significant economic fluctuations. High variance in these indicators often boosts the value of SVM, which handles complex boundaries.
	3.	**Improved Gains with Tuning**: The SVM’s jump from 79.63% to 98.77% test accuracy underscores how **crucial** hyperparameter tuning can be—particularly for advanced algorithms like SVM.
	4.	**Scalability**: 

---
#Module17
