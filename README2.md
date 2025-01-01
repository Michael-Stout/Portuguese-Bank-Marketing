**Executive Summary**
---
This project aims to increase long-term deposit accounts through advanced predictive analytics. I analyzed customer behavior patterns and campaign effectiveness by leveraging machine-learning techniques to optimize conversion rates.

- **Data**: 4,119 examples, 21 features, 1,230 missing entries.
- **Best Model**: Tuned SVM with 98.77% test accuracy and 0.9973 ROC AUC.
- **Decision Tree**: Second place (94.55%), very fast but prone to overfitting.
- **KNN**: 90.94% test accuracy, also near-perfect fit on training.
- **Logistic Regression**: 74.66% accuracy, stable and interpretable, but lags behind tree/kernel-based models.

**Recommendation**

Utilize SVM for best performance or Decision Tree for speed with minimal performance drop.

While SVM is the champion, it’s also the slowest. Running the code on the larger version of the dataset was canceled after over 12 hours of processing and would require distributed training or more computational resources to complete.

Key Features of My Models

**Data Description**
---
In this analysis, I explore how to predict potential subscriptions to a long-term deposit product using data from a Portuguese bank marketing campaign.

The pipeline involves data cleaning, exploratory data analysis (EDA), various classification models, and hyperparameter tuning.

**Data Description**
- **Shape**: (4119, 21)
- **Null Values**: 1230 missing entries
- **Column Data Types**:
	- 11 object (categorical)
	- 5 int64 (integer)
   	- 5 float64 (numeric/floating)

**Numerical Features**
1. age: Age of the customer
2. duration: Last contact duration in seconds
3. campaign: Number of contacts performed during this campaign	
4. pdays: Number of days since the client was last contacted
5. previous: Number of contacts performed before this campaign
6. emp.var.rate: Employment variation rate
7. cons.price.idx: Consumer price index
8. cons.conf.idx: Consumer confidence index
9. euribor3m: Euribor 3-month rate
10. nr.employed: Number of employees

**Categorical Features**:
1. job: Type of job
2. marital: Marital status
3. education: Education level
4. default: Is credit in default?
5. housing: Has a housing loan?
6. loan: Has a personal loan?
7. contact: Type of communication contact
8. month: Last contact month
9. day_of_week: Last contact day of the week
10. poutcome: The outcome of the previous marketing campaign

**Exploratory Data Analysis (EDA)**
---

**Summary**
* 	**Shape:** There are 4119 rows and 21 columns
* 	**Missingness**: There are 1,230 "unknown" values spread mostly in default, housing, loan, and some education.

**Numerical Stats**:
* 	Average age is ~40, ranging 18–88.
* 	duration (excluded for real modeling) was originally up to ~3,600 seconds.
* 	Macro-Economic Cluster: euribor3m & emp.var.rate strongly correlates with economic conditions.

**Target Variable**: For the raw data, ~11% “yes” to term deposit vs. ~89% “no,” but an upsampling approach balanced classes to 50/50 in the training set.

**Summary of Notable Distributions**
* 	**job**: ~24.57% admin., 21.46% blue-collar, 16.78% technician, etc.
* 	**housing**: ~52.80% yes, ~44.65% no, ~2.55% missing (NaN)
* 	**loan**: ~81.31% no, ~16.14% yes, ~2.55% missing
  
**Supporting Visualization and Stats**

You may view all visualizations for your own analysis [here.](/output "All Plots")

## **Numerical Features**

### **Age of the customer**

![Age](output/num_age_distribution.png "Age")

**Descriptive Statistics for age:**
{'count': 4119.0, 'mean': 40.11361981063365, 'std': 10.313361547199813, 'min': 18.0, '25%': 32.0, '50%': 38.0, '75%': 47.0, 'max': 88.0}

**Shape:** Right-skewed, with most ages clustering between the late 20s and early 40s.

**Implication:** The population skews younger to middle-aged, but there are still some older customers in the dataset.

### **Last contact duration in seconds**

![Age](output/num_duration_distribution.png "Duration")

**Descriptive Statistics for duration:**
{'count': 4119.0, 'mean': 256.7880553532411, 'std': 254.70373612073678, 'min': 0.0, '25%': 

**Shape:** Strongly right-skewed, with a large majority of call durations under ~500 seconds and a very long tail of longer calls.

**Implication:** Most calls are relatively short, but a small number can last much longer.

### **Correlation Matrix**

![Age](output/correlation_matrix.png "Campaign")

**Macro-Economic Cluster**
*	emp.var.rate, euribor3m, and nr.employed are all strongly positively correlated (in the 0.9+ range).
*	cons.price.idx also has a notable positive correlation with these variables (around 0.76–0.97).

**Implication:** These four features (employment variation rate, consumer price index, euribor 3-month rate, and number of employees) often move together, likely reflecting broad economic conditions.

### **Target Variable**

![Class Distribution](output/class_distribution.png "lass Distribution")

**Bar Chart:** Visualizes the raw counts of customers who did not subscribe (no) versus those who did subscribe (yes) to a term deposit. Here, around 3,668 records (89.05%) are labeled “no,” while 451 records (10.95%) are labeled “yes.”

**Pie Chart:** Shows the same data as percentages, highlighting a class imbalance in the target variable—about 11% of customers ended up subscribing, whereas nearly 89% did not.

**Implication:** This imbalanced class distribution may require special techniques (e.g., class weighting, oversampling, or undersampling) to train fair and robust models. Simply predicting “no” for everyone would achieve ~89% accuracy, so more nuanced methods are needed to improve predictions for the minority (“yes”) class.

## **Categorical Features**

### **Job versus Target**

![Job](output/relationship_job_target.png "Job")

**Overall:** Most customers in each job category choose “no,” reflecting the dataset’s general imbalance.

**High Counts:** Blue-collar and admin. jobs appear most common, but both also have relatively fewer “yes” conversions proportionally.

**Smaller Categories:** Retired and student groups, though smaller, seem to have a notable fraction of “yes,” suggesting these groups might be more receptive.

**Implication:** Different occupations show varying interest levels in term deposits; certain smaller groups (like retirees) might convert at higher rates despite lower overall counts.

### **Housing versus Target**

![Housing](output/relationship_housing_target.png "Housing")

**Housing “yes”:** Most customers with a “yes” for housing also say “no” to the product, but a moderate segment does say “yes.”

**Housing “no”:** Slightly fewer overall customers than “yes” for housing, but there’s still a substantial number of “no” responses.

**Unknown:** Very small group; may need imputation or separate analysis.

**Implication:** Having a housing loan does not strictly prevent or guarantee interest in term deposits, but there could be subtle differences in conversion rates between those who have (or don’t have) housing loans.

### Loan versus Target

![Loan](output/relationship_loan_target.png "Loan")

**Loan “no”:** This is the largest group; though many are “no” to the product, it also contains a meaningful fraction of “yes.”

**Loan “yes”:** Fewer total observations, but similarly skewed toward “no.”

**Unknown:** Minimal count, potentially “unknown” or missing data on personal loans.

**Implication:** Whether someone has a personal loan does not appear as decisive as the housing category, but still might combine with other factors (income, age, job) to influence deposit subscriptions.

## **Modeling Approaches**

I used four classifiers:
1.	**Logistic Regression**
2.	**Decision Tree**
3.	**K-Nearest Neighbors (KNN)**
4.	**Support Vector Machine (SVM)**

**Imbalanced Data**
Before training, I upsampled the minority class to handle the imbalance. Then, I split data (80/20) into train/test sets.

**Evaluation Metrics**
* 	Accuracy
* 	ROC AUC
* 	Precision, Recall, F1 (seen in classification reports)

**Default Results**

I obtained the following results using the default settings for the models.

| Model               | Train Time (s) | Train Acc | Test Acc | Precision | Recall | F1 Score |
|---------------------|----------------|-----------|----------|-----------|---------|-----------|
| Logistic Regression | 0.26           | 0.7384    | 0.7466   | 0.8187    | 0.6335  | 0.7143    |
| Decision Tree       | 0.02           | 0.9997    | 0.9455   | 0.9017    | 1.0000  | 0.9483    |
| KNN                 | 0.00           | 0.9008    | 0.8590   | 0.7912    | 0.9755  | 0.8737    |
| SVM                 | 2.19           | 0.8062    | 0.7963   | 0.8560    | 0.7125  | 0.7777    |

**Observations**:
1. Decision Tree Overfitting
* It achieves a near-perfect training accuracy (99.97%) but still shows strong generalization on the test set (94.55%).
* Its recall is a perfect 1.0, indicating it identifies all positive cases in the test set—however, this can hint that the tree is very deep or complex.
2. KNN’s Speed vs. Performance
* KNN trains extremely quickly (0.00s reported) yet yields a respectable test accuracy of 85.90%.
* Its recall (0.9755) is also quite high, meaning it rarely misses positive cases, although precision is lower than other models.
3. SVM’s Balance and Train Time
* SVM offers a decent balance (79.63% test accuracy, 0.856 precision, 0.7125 recall), but it takes considerably longer to train (2.19s) than the other methods.
* This might be acceptable if the additional compute time is not a bottleneck, depending on the application.
4. Logistic Regression’s Interpretability
* Logistic Regression is relatively fast (0.26s) with moderate train and test accuracies around ~74%.
* Precision (0.8187) and recall (0.6335) are decent, suggesting a balanced, interpretable baseline solution.
5. Comparative Precision and Recall
* Decision Tree is the clear outlier, combining high precision (0.9017) and perfect recall on the test set, leading to the top F1 score (0.9483).
* KNN has the highest recall (0.9755) but a lower precision (0.7912), reflecting more false positives.

Overall, the Decision Tree dominates in accuracy and recall, but the near-perfect training score suggests potential overfitting. KNN and SVM strike a more balanced compromise, while Logistic Regression remains a solid, interpretable baseline with lower computational cost.

**Hyperparameter Tuning Results**

The following results were achieved after hyperparameter tuning.

| Model | Time (s) | Train Acc | Test Acc | ROC AUC | Best Params |
|-------|----------|-----------|-----------|----------|-------------|
| Logistic Regression | 8.01 | 0.739093 | 0.746594 | 0.796585 | {'C': 10, 'l1_ratio': 0.9, 'max_iter': 1000, 'penalty': 'elasticnet', 'solver': 'saga'} |
| Decision Tree | 0.58 | 0.999659 | 0.945504 | 0.945812 | {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2} |
| KNN | 0.20 | 0.999659 | 0.909401 | 0.948080 | {'n_neighbors': 3, 'weights': 'distance'} |
| SVM | 55.68 | 0.997955 | 0.987738 | 0.997275 | {'C': 10, 'gamma': 1, 'kernel': 'rbf'} |



**Findings**:

Here's the consolidated model performance analysis:

## Model Performance Summary

| Model | Test Accuracy | Notable Characteristics |
|-------|---------------|------------------------|
| SVM | 98.77% | Best ROC AUC (0.9973), Longest training time (60+ seconds) |
| Decision Tree | 94.55% | Fast training, Shows overfitting |
| KNN | 90.94% | Perfect training accuracy but lower test performance |
| Logistic Regression | 74.66% | Most interpretable, Fastest training |

## Dataset Characteristics

The analysis was conducted on 4,119 records with 1,230 null values, requiring careful imputation, particularly for default, housing, loan, and education fields. The original class distribution showed a significant skew toward "no" responses, addressed through upsampling in the training set.

## Key Insights

The Support Vector Machine (SVM) emerged as the superior model, achieving near-perfect performance despite longer training times. While the Decision Tree and KNN showed strong performance, they exhibited signs of overfitting. Though less accurate, Logistic Regression offers advantages in interpretability and training speed.

**Further Work & Considerations**
1. **Use SVM** if maximum predictive power is desired and computation time is acceptable.
2. **Watch for Overfitting** in the Decision Tree and KNN; while these show strong test performance, they can degrade if future data differs from the training set.
3. **Focus on Data Quality**: 1,230 missing entries are significant; refining or collecting missing info could further improve results.
4. **Leverage Economic Indicators** (emp.var.rate, euribor3m) and call features (e.g., duration) more explicitly—SVM permutations indicate these are key.
5. **Iterate** with advanced feature engineering, model ensembles, or time-series analysis for additional gains.