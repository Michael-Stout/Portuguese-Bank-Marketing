# **Executive Summary**
---
This project aims to increase long-term deposit accounts through advanced predictive analytics. I analyzed customer behavior patterns and campaign effectiveness by leveraging machine-learning techniques to optimize conversion rates.

- **Data**: 4,119 examples, 21 features, 1,230 missing entries.
- **Best Model**: Tuned SVM with **98.77%** test accuracy and **0.9973** ROC AUC.
- **Decision Tree**: Second place (**94.55%**), very fast but prone to overfitting.
- **KNN**: **90.94%** test accuracy, also near-perfect fit on training.
- **Logistic Regression**: **74.66%** accuracy, stable and interpretable, but lags behind tree/kernel-based models.

## **Recommendation**
Utilize **SVM** for best performance or **Decision Tree** for speed with minimal performance drop.  
While SVM is the champion, it’s also the slowest. Running the code on the larger dataset was canceled after over 12 hours of processing and would require distributed training or more computational resources to complete.

---

# **Data Description**
In this analysis, I explore how to predict potential subscriptions to a long-term deposit product using data from a Portuguese bank marketing campaign.  
The pipeline involves data cleaning, exploratory data analysis (EDA), various classification models, and hyperparameter tuning.

### **Dataset Overview**
- **Shape**: (4119, 21)  
- **Null Values**: 1230 missing entries  
- **Column Data Types**:  
  - 11 object (categorical)  
  - 5 int64 (integer)  
  - 5 float64 (numeric/floating)  

### **Numerical Features**  
1. `age`: Age of the customer  
2. `duration`: Last contact duration in seconds  
3. `campaign`: Number of contacts performed during this campaign  
4. `pdays`: Number of days since the client was last contacted  
5. `previous`: Number of contacts performed before this campaign  
6. `emp.var.rate`: Employment variation rate  
7. `cons.price.idx`: Consumer price index  
8. `cons.conf.idx`: Consumer confidence index  
9. `euribor3m`: Euribor 3-month rate  
10. `nr.employed`: Number of employees  

### **Categorical Features**  
1. `job`: Type of job  
2. `marital`: Marital status  
3. `education`: Education level  
4. `default`: Is credit in default?  
5. `housing`: Has a housing loan?  
6. `loan`: Has a personal loan?  
7. `contact`: Type of communication contact  
8. `month`: Last contact month  
9. `day_of_week`: Last contact day of the week  
10. `poutcome`: Outcome of the previous marketing campaign  

---

# **Exploratory Data Analysis (EDA)**

## **Class Distribution**
![Class Distribution (Counts & Percentages)](output/class_distribution.png "Class Distribution")

- **Counts**: 3,668 labeled “no” (89.05%) vs. 451 labeled “yes” (10.95%).  
- **Implication**: There is a class imbalance, so special techniques (e.g., oversampling or class weights) may be necessary.  

## **Correlation Matrix**
![Hierarchically Clustered Correlation Matrix](output/correlation_matrix_clustered.png "Correlation Matrix")

**Macro-Economic Cluster**  
- `emp.var.rate`, `euribor3m`, and `nr.employed` are all strongly positively correlated (in the 0.9+ range).  
- `cons.price.idx` is also notably correlated with these variables (~0.76–0.97).  
- `pdays` has an inverse correlation with `previous` (-0.59).  

**Implication**  
These clusters of features often move together, reflecting broad economic conditions. Proper modeling or feature engineering may consider these interdependencies.

---

## **Selected Categorical Features**

### **Job vs. Target**
![Job vs. Target](output/relationship_job_target.png "Job vs. Target")

- **High Counts**: Blue-collar, admin., and technician appear most frequently.  
- **Conversion**: Retirees and students, while fewer in number, seem to have relatively higher “yes” rates.

### **Housing vs. Target**
![Housing vs. Target](output/relationship_housing_target.png "Housing vs. Target")

- **Breakdown**: ~52.80% “yes” to housing loan, ~44.65% “no.”  
- **Observation**: No strict relationship with deposit subscription alone, but it can still play a role in combination with other financial indicators.

### **Loan vs. Target**
![Loan vs. Target](output/relationship_loan_target.png "Loan vs. Target")

- **Loan “no”**: Largest group; also contains the most “yes” conversions by count.  
- **Loan “yes”**: Smaller overall but includes both “no” and “yes” to the product.

---

# **Modeling Approaches**
Four classifiers were chosen to predict deposit subscription (`yes`/`no`):
1. **Logistic Regression**  
2. **Decision Tree**  
3. **K-Nearest Neighbors (KNN)**  
4. **Support Vector Machine (SVM)**  

### **Imbalanced Data Handling**
- **Upsampling**: The minority class was oversampled to address class imbalance.  
- **Train/Test Split**: 80% training data, 20% test data.

### **Evaluation Metrics**
- **Accuracy**  
- **ROC AUC**  
- **Precision, Recall, F1** (from classification reports)

---

## **Default Model Results**
| Model               | Train Time (s) | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|:--------------------|:--------------:|:--------------:|:-------------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.08           | 0.7384         | 0.7466        | 0.8187    | 0.6335 | 0.7143   |
| Decision Tree       | 0.02           | 0.9997         | 0.9455        | 0.9017    | 1.0000 | 0.9483   |
| KNN                 | 0.00           | 0.9008         | 0.8590        | 0.7912    | 0.9755 | 0.8737   |
| SVM                 | 2.18           | 0.8062         | 0.7963        | 0.8560    | 0.7125 | 0.7777   |

**Observations**  
- **Decision Tree**: Near-perfect train accuracy, test accuracy of 0.9455, perfect recall (1.0) → likely overfitting.  
- **KNN**: Quick to train (0.00s), good recall (0.9755), test accuracy 0.8590.  
- **SVM**: Balanced but slower, 0.7963 test accuracy.  
- **Logistic Regression**: Fastest training, interpretability advantage, 0.7466 test accuracy.

---

## **Hyperparameter Tuning Results**
| Model               | Time (s) | Train Accuracy | Test Accuracy | ROC AUC  | Best Params                                                                             |
|:--------------------|:--------:|:-------------:|:-------------:|:--------:|:----------------------------------------------------------------------------------------|
| Logistic Regression | 8.62     | 0.7391        | 0.7466        | 0.7966   | {'C': 10, 'l1_ratio': 0.9, 'max_iter': 1000, 'penalty': 'elasticnet', 'solver': 'saga'} |
| Decision Tree       | 0.52     | 0.9997        | 0.9455        | 0.9458   | {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}                      |
| KNN                 | 0.22     | 0.9997        | 0.9094        | 0.9481   | {'n_neighbors': 3, 'weights': 'distance'}                                               |
| SVM                 | 58.04    | 0.9980        | 0.9877        | 0.9973   | {'C': 10, 'gamma': 1, 'kernel': 'rbf'}                                                  |

## **Model Performance Summary**
| Model             | Test Accuracy | Notable Characteristics                           |
|:------------------|:------------:|:--------------------------------------------------|
| **SVM**           | **98.77%**   | Best overall; extremely accurate; slow to train   |
| **Decision Tree** | 94.55%       | Very fast; overfits with perfect train accuracy   |
| **KNN**           | 90.94%       | Near-perfect train accuracy; decent generalization|
| **Logistic Reg.** | 74.66%       | Interpretable, stable, but lower accuracy         |

---

## **Key Insights**
1. **SVM Dominance**  
   - Achieves top test accuracy (98.77%) and highest ROC AUC (0.9973).  
   - Training can be very slow on large datasets.

2. **Decision Tree & KNN**  
   - Excellent performance (94.55%, 90.94% test accuracy).  
   - Overfitting is a concern (both ~99.97% train accuracy).

3. **Logistic Regression**  
   - Most interpretable with moderate performance.  
   - Fastest training time.

4. **Data Quality**  
   - 1,230 missing entries is significant.  
   - Better imputation or data collection may further improve performance.

5. **Feature Importance**  
   - Economic indicators (`emp.var.rate`, `euribor3m`, `nr.employed`) and `cons.price.idx` are strong predictors.  
   - `age`, `campaign`, `pdays`, and `previous` also matter.

---

## **Further Work & Considerations**
1. **Use SVM** for maximum predictive power if resources allow.  
2. **Decision Tree** is a good alternative for speed, but watch for overfitting.  
3. **Improve Data Quality**: Address missing values in `default`, `loan`, `housing`, `education`, etc.  
4. **Feature Engineering**: Combine or transform existing variables (e.g., create age groups).  
5. **Ensemble Methods** (e.g., Random Forest, XGBoost) often beat single models.  
6. **Periodic Retraining** with updated data is important, especially for economic-related features.

---
# **Tuned Models and Their Visualizations**

Below are the final, tuned versions of our models along with key plots (permutation feature importance, partial dependence plots, ROC curve, and precision–recall curve). These help us understand **how** each model arrives at its predictions and **why** certain features matter more than others.

## **SVM (Tuned)**

The **SVM** model achieved the highest accuracy (98.77%) and ROC AUC (0.9973) in our experiments.

### **1. Permutation Feature Importance**

![Permutation Feature Importance - SVM (Tuned)](output/permutation_importance_SVM_Tuned.png "Permutation Feature Importance (SVM)")

- **Interpretation**:  
  - The top features for SVM include `nr.employed`, `age`, `cons.conf.idx`, and `cons.price.idx`—all strong indicators of economic and demographic conditions.  
  - Categorical encodings such as `marital_married`, `housing_yes`, and `education_university.degree` also appear highly important.  
  - Higher bars indicate greater impact on the model's predictions, and the black lines represent the standard deviation from multiple permutations.

### **2. Partial Dependence Plots (PDPs)**

Partial dependence plots illustrate how changes in a single feature affect the model’s predicted probability of a positive outcome (`y = yes`), holding all other features constant.

1. **`emp.var.rate`**  
   ![Partial Dependence for 'emp.var.rate' (SVM)](output/pdp_SVM_Tuned_emp.var.rate.png "emp.var.rate PDP")
   - **Interpretation**:  
     - The model predicts higher probabilities of subscription when `emp.var.rate` is in a moderate negative range (~ -1.0 to -0.5).  
     - Very negative or very positive values reduce the predicted likelihood of a positive outcome.  

2. **`nr.employed`**  
   ![Partial Dependence for 'nr.employed' (SVM)](output/pdp_SVM_Tuned_nr.employed.png "nr.employed PDP")
   - **Interpretation**:  
     - Slight dips in `nr.employed` around 0 to -0.5 lead to lower probabilities, while more negative or moderately positive values can push probabilities higher.  

3. **`age`**  
   ![Partial Dependence for 'age' (SVM)](output/pdp_SVM_Tuned_age.png "age PDP")
   - **Interpretation**:  
     - Ages roughly in the mid-30s to early 40s correlate with the highest predicted probability of subscription.  
     - Probability drops off substantially for older ages.

4. **`cons.conf.idx`**  
   ![Partial Dependence for 'cons.conf.idx' (SVM)](output/pdp_SVM_Tuned_cons.conf.idx.png "cons.conf.idx PDP")
   - **Interpretation**:  
     - Moderate consumer confidence indexes (around 0 to 1) are linked to higher probabilities of subscription; extremes at either end reduce the likelihood.

5. **`cons.price.idx`**  
   ![Partial Dependence for 'cons.price.idx' (SVM)](output/pdp_SVM_Tuned_cons.price.idx.png "cons.price.idx PDP")
   - **Interpretation**:  
     - The plot shows a bimodal region (roughly -1.0 to -0.5 and +0.5 to +1.0) where consumer price index correlates with higher subscription likelihood.  
     - Very low or very high indices drop the predicted probability significantly.

### **3. ROC Curve**

![ROC Curve for SVM (Tuned)](output/roc_curve_SVM_Tuned.png "ROC Curve - SVM (Tuned)")

- **Interpretation**:  
  - The nearly perfect curve indicates the model excels at distinguishing positive vs. negative classes.  
  - **AUC = 0.9973** confirms outstanding separability of the classes.

### **4. Precision–Recall Curve**

![Precision–Recall Curve for SVM (Tuned)](output/precision_recall_SVM_Tuned.png "Precision–Recall Curve - SVM (Tuned)")

- **Interpretation**:  
  - The curve remains high (>0.90 precision) over a broad range of recall values.  
  - **AP ~0.9853** underscores the model’s strong ability to capture positive cases while keeping false positives low.

---

## **Summary of SVM (Tuned) Visualizations**
- **Feature Importance** reveals that economic indicators (`nr.employed`, `emp.var.rate`, etc.) and demographics (`age`) strongly influence the SVM’s decisions.  
- **Partial Dependence** plots confirm that mid-range economic indicators and specific age ranges correspond to higher subscription likelihood.  
- **ROC** and **Precision–Recall** curves show the SVM to be **extremely effective**, albeit with higher computational costs.

*(End of Full Report)*  
