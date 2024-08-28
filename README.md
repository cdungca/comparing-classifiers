# comparing-classifiers

A study comparing the performance of the following classifiers: logistic regression, k-nearest neighbors, decision trees, and support vector machines. This is part of the course work for [UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence](https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence). 

CRISP-DM^1^ methodology was adopted in this study. The data was taken from [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The dataset is related to 17 marketing campaigns from a Portuguese banking institution.

![alt text](https://en.wikipedia.org/wiki/File:CRISP-DM_Process_Diagram.png "Image from Wikipedia")

You can follow the detailed analysis done in the [Jupyter notebook](https://github.com/cdungca/comparing-classifiers/blob/main/prompt_III.ipynb).

## Objective

The main objective of this study is to compare the performance of the different classifiers (logistic regression, k-nearest neighbors, decision trees, and support vector machines) and identify the best suited for a predictive model. This model will be used to identify the features and characteristics to increase campaign success.   

## Data Analysis

Here are the fields included in the dataset, bank-additional-full.csv

1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')
8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')

To start the analysis, we've looked at the data distribution on the categorical fields and performed pre-processing or cleaning of data. Here are some of the data distributions:

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/age_distribution.png "Age Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/campaign_distribution.png "Campaign Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/contact_distribution.png "Contact Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/default_distribution.png "Default Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/housing_distribution.png "Housing Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/loan_distribution.png "Loan Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/marital_distribution.png "Marital Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/default_distribution.png "Poutcome Distribution")

Since our goal it choose a classifier for predictive model, we will remove duration since this feature highly affects the target (e.g., if duration=0 then y='no').

In pdays numeric field, client not previously contacted contains a value of 999. This will affect the scaling of data and we should just replace it to 0. 

The following categorical fields will be converted to numeric using One Hot Encoding:

- job
- marital
- education
- default
- housing
- loan
- contact
- month
- day_of_week
- poutcome

The following numeric fields will be scaled:

- age
- campaign
- pdays
- previous
- emp.var.rate
- cons.proce.idx
- cons.conf.idx
- euribor3m

Here's the table comparing the performance of the different classifiers using the default parameters:

|model|train score|test score|ave fit time (sec)|
|--|--|--|--|
|Logistic Regression| 0.900230|0.896475|1.340354|
|KNN|0.914053|0.887054|1.089069|
|Decision Tree|0.995047|0.840342|2.749162|
|SVM|0.906057|0.900845|354.994642|

And here are the Confusion Matrix and Roc Curve for each classifier:

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/lgr_default-cmroc.png "Logistic Regression - Confusion Matrix and ROC Curve")
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/knn_default-cmroc.png "KNN - Confusion Matrix and ROC Curve")
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/dtree_default-cmroc.png "Decision Tree - Confusion Matrix and ROC Curve")
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/svm_default-cmroc.png "SVM - Confusion Matrix and ROC Curve")

## Recommendation



## References

1. Shearer C., The CRISP-DM model: the new blueprint for data mining, J Data Warehousing (2000); 5:13—22.






