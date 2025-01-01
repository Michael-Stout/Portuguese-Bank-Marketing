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
While SVM is the champion, it’s also the slowest. Running the code on the larger version of the dataset was canceled after over 12 hours of processing and would require distributed training or more computational resources to complete.

## **Key Features of My Models**
In multiple classification approaches, the following features stand out as highly predictive for term deposit subscription:
- **emp.var.rate** (employment variation rate)
- **euribor3m** (3-month Euribor rate)
- **cons.conf.idx** (consumer confidence index)
- **cons.price.idx** (consumer price index)
- **nr.employed** (number of employees)
- **age** (customer age)
- **pdays** (days since the client was last contacted)

---

# **Data Description**
---
In this analysis, I explore how to predict potential subscriptions to a long-term deposit product using data from a Portuguese bank marketing campaign.  
The pipeline involves data cleaning, exploratory data analysis (EDA), various classification models, and hyperparameter tuning.

### **Dataset Characteristics**

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
---
### **Summary**
- **Shape**: 4119 rows and 21 columns  
- **Missingness**: 1,230 "unknown" (NaN) values spread mostly in `default`, `housing`, `loan`, and some `education`.  
- **Class Imbalance**: ~11% “yes” vs. ~89% “no” for term deposits. An upsampling approach balanced classes to 50/50 in the training set.

### **Numerical Stats**
- Average age is ~40, ranging 18–88.
- `duration` (often excluded in real modeling to avoid data leakage) can range up to ~3,600 seconds.
- Macro-Economic Cluster: `euribor3m` and `emp.var.rate` strongly correlate with economic conditions.

### **Distribution Examples**

#### **Age of the Customer**

laskdjfalskfjalsjfkklasdjfasf

- ~81.31% do not have a personal loan, ~16.14% do, ~2.55% missing.
- Subtle differences in conversion rates but not a primary factor alone.

---

# **Modeling Approaches**
I used four classifiers to predict the “yes”/”no” subscription:
1. **Logistic Regression**  
2. **Decision Tree**  
3. **K-Nearest Neighbors (KNN)**  
4. **Support Vector Machine (SVM)**  

### **Imbalanced Data Handling**
- **Upsampling** was applied to the minority class to address imbalance.  
- Data was split 80%/20% into train/test sets.  

### **Evaluation Metrics**
- **Accuracy**  
- **ROC AUC**  
- **Precision, Recall, F1** (seen in classification reports)

---

## **Default Results**
| Model               | Train Time (s) | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|:--------------------|:--------------:|:--------------:|:-------------:|:---------:|:------:|:--------:|
| Logistic Regression | 0.08           | 0.7384         | 0.7466        | 0.8187    | 0.6335 | 0.7143   |
| Decision Tree       | 0.02           | 0.9997         | 0.9455        | 0.9017    | 1.0000 | 0.9483   |
| KNN                 | 0.00           | 0.9008         | 0.8590        | 0.7912    | 0.9755 | 0.8737   |
| SVM                 | 2.18           | 0.8062         | 0.7963        | 0.8560    | 0.7125 | 0.7777   |

**Observations**:
1. **Decision Tree**:  
   - Near-perfect train accuracy (99.97%) but still strong test accuracy (94.55%).  
   - Perfect recall (1.0), indicating it finds all positives. Overfitting is likely.  
2. **KNN**:  
   - Trains extremely fast, yields 85.90% test accuracy.  
   - Very high recall (0.9755), but moderate precision (0.7912).  
3. **SVM**:  
   - Decent accuracy (79.63%), takes longer to train (2.18s).  
   - Good balance of precision (0.8560) and recall (0.7125).  
4. **Logistic Regression**:  
   - Fast (0.08s), moderate performance (74.66% test accuracy).  
   - Balanced precision (0.8187) vs. recall (0.6335).

---

## **Hyperparameter Tuning Results**

| Model               | Time (s) | Train Accuracy | Test Accuracy | ROC AUC  | Best Params                                                                             |
|:--------------------|:-------:|:-------------:|:-------------:|:--------:|:----------------------------------------------------------------------------------------|
| Logistic Regression | 8.62    | 0.7391        | 0.7466        | 0.7966   | {'C': 10, 'l1_ratio': 0.9, 'max_iter': 1000, 'penalty': 'elasticnet', 'solver': 'saga'} |
| Decision Tree       | 0.52    | 0.9997        | 0.9455        | 0.9458   | {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2}                      |
| KNN                 | 0.22    | 0.9997        | 0.9094        | 0.9481   | {'n_neighbors': 3, 'weights': 'distance'}                                               |
| SVM                 | 58.04   | 0.9980        | 0.9877        | 0.9973   | {'C': 10, 'gamma': 1, 'kernel': 'rbf'}                                                  |

---

## **Model Performance Summary**
| Model               | Test Accuracy | Notable Characteristics                                           |
|:--------------------|:------------:|:------------------------------------------------------------------|
| **SVM**             | **98.77%**   | Best ROC AUC (0.9973), longest training time (~60s)               |
| **Decision Tree**   | 94.55%       | Very fast training, overfits (perfect or near-perfect train acc.) |
| **KNN**             | 90.94%       | Near-perfect train acc., moderate test performance                |
| **Logistic Reg.**   | 74.66%       | Most interpretable, stable, faster training                       |

---

## **Key Insights**
1. **SVM Dominance**  
   - Achieves the best test accuracy (98.77%) and highest ROC AUC (0.9973).  
   - Expensive computationally and slow to train at scale.  

2. **Decision Tree & KNN**  
   - Both yield high accuracy (94.55% and 90.94%, respectively).  
   - Show signs of overfitting (almost perfect on training set).  
   - Decision Tree is extremely fast, but may degrade if future data distribution changes.  

3. **Logistic Regression**  
   - Simplest and most interpretable.  
   - Lower accuracy (74.66%) but good for baseline or resource-limited scenarios.

4. **Data Quality**  
   - 1,230 missing (or unknown) entries is significant.  
   - Addressing data gaps and refining imputation may boost model performance.

5. **Feature Importance**  
   - Features like `emp.var.rate`, `cons.price.idx`, `euribor3m`, and `nr.employed` (economic indicators) heavily influence outcome.  
   - `age`, `campaign`, `pdays`, and `previous` are also relevant.

---

## **Further Work & Considerations**
1. **Use SVM** for maximum predictive power if computation time and resources permit.  
2. **Decision Tree** can be a good alternative when speed matters, but watch for overfitting.  
3. **Continue Data Collection**: Resolve missing values (especially in `default`, `loan`, `housing`, `education`).  
4. **Feature Engineering**: Combine or transform features (e.g., grouping age ranges, analyzing call durations more granularly).  
5. **Ensemble Methods**: Explore ensembles (e.g., Random Forest, XGBoost) that may outperform single-model approaches.  
6. **Time-Series or Economic Shifts**: Economic variables can change significantly over time, so periodic retraining is advised.
