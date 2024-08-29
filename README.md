# comparing-classifiers

A study comparing the performance of the following classifiers: logistic regression, k-nearest neighbors, decision trees, and support vector machines. This is part of the course work for [UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence](https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence). 

CRISP-DM[^1] methodology was adopted in this study. The data was taken from [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The dataset is related to 17 marketing campaigns from a Portuguese banking institution.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/598px-CRISP-DM_Process_Diagram.png "Image from Wikipedia")

You can follow the detailed analysis done in the [Jupyter notebook](https://github.com/cdungca/comparing-classifiers/blob/main/prompt_III.ipynb).

## Objective

The main objective of this study is to compare the performance of the different classifiers (logistic regression, k-nearest neighbors, decision trees, and support vector machines) and identify the best suited for a predictive model. This model will be used to identify the features and characteristics to increase campaign success.   

## Data Analysis

Here are the fields included in the dataset, bank-additional-full.csv

|Field|Description|
|--|--|
|age|(numeric)|
|job|type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')|
|marital|marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)|
|education|(categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')|
|default|has credit in default? (categorical: 'no','yes','unknown')|
|housing|has housing loan? (categorical: 'no','yes','unknown')|
|loan|has personal loan? (categorical: 'no','yes','unknown')|
|contact|contact communication type (categorical: 'cellular','telephone')|
|month|last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')|
|day_of_week|last contact day of the week (categorical: 'mon','tue','wed','thu','fri')|
|duration|last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.|
|campaign|number of contacts performed during this campaign and for this client (numeric, includes last contact)|
|pdays|number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)|
|previous|number of contacts performed before this campaign and for this client (numeric)|
|poutcome|outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')|
|emp.var.rate|employment variation rate - quarterly indicator (numeric)|
|cons.price.idx|consumer price index - monthly indicator (numeric)|
|cons.conf.idx|consumer confidence index - monthly indicator (numeric)|
|euribor3m|euribor 3 month rate - daily indicator (numeric)|
|nr.employed|number of employees - quarterly indicator (numeric)|
|y|has the client subscribed a term deposit? (binary: 'yes','no')|

Target field in the dataset is y.

To start the analysis, we've looked at the data distribution on the categorical fields and performed pre-processing or cleaning of data. 

Here are some of the data distributions:

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/age_distribution.png "Age Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/campaign_distribution.png "Campaign Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/contact_distribution.png "Contact Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/default_distribution.png "Default Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/housing_distribution.png "Housing Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/loan_distribution.png "Loan Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/marital_distribution.png "Marital Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/poutcome_distribution.png "Poutcome Distribution")

Since our goal is to choose a classifier for predictive model, we will remove duration since this feature highly affects the target (e.g., if duration=0 then y='no').

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

|Model|Train Accuracy|Test Accuracy|Test Precision|Fit Time (sec)|
|--|--|--|--|--|
|Logistic Regression|0.900359|0.896572|0.617391	|1.219468|
|KNN|0.914020|0.887152|0.498361|1.167804|
|Decision Tree|0.995047|0.838302|0.299444|1.534464|
|SVM|0.906057|0.900845|**0.674185**|195.869979|

|Model|Train Accuracy|Test Accuracy|Test Precision|Fit Time (sec)|
|--|--|--|--|--|
|Logistic Regression|0.899485|0.895212|0.625995|1.268780|
|KNN|0.924962|0.882490|0.464689|1.196504|
|Decision Tree|0.902593|0.900748|0.661972|1.289213|
|SVM|0.900165|0.897155|**0.675958**|113.831252|

And here are the Confusion Matrix and Roc Curve for each classifier:

Using Default Parameters

Logistic Regression (default)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/lgr_default-cmroc.png "Logistic Regression - Confusion Matrix and ROC Curve")

KNN (default)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/knn_default-cmroc.png "KNN - Confusion Matrix and ROC Curve")

Decision Tree (default)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/dtree_default-cmroc.png "Decision Tree - Confusion Matrix and ROC Curve")

SVM (default)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/svm_default-cmroc.png "SVM - Confusion Matrix and ROC Curve")

Using Best Parameters

Logistic Regression (C=0.01, Solver = liblinear)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/lgr_best-cmroc.png "Logistic Regression - Confusion Matrix and ROC Curve")

KNN (n_neighbors = 3, weights = uniform)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/knn_best-cmroc.png "KNN - Confusion Matrix and ROC Curve")

Decision Tree (max_depth = 5, min_samples_leaf = 2, criterion = gini)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/dtree_best-cmroc.png "Decision Tree - Confusion Matrix and ROC Curve")

SVM (c= 0.01)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/svm_best-cmroc.png "SVM - Confusion Matrix and ROC Curve")


## Recommendation




[^1]: Shearer C., The CRISP-DM model: the new blueprint for data mining, J Data Warehousing (2000); 5:13—22.






