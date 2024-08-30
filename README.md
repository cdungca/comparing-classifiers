# comparing-classifiers

A study comparing the performance of the different classifiers: logistic regression, k-nearest neighbors, decision trees, and support vector machines. This is part of the course work for [UC Berkeley Professional Certificate in Machine Learning and Artificial Intelligence](https://em-executive.berkeley.edu/professional-certificate-machine-learning-artificial-intelligence). 

CRISP-DM[^1] methodology was adopted in this study and the data was taken from [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing). The dataset is related to 17 marketing campaigns from a Portuguese banking institution.

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/CRISP-DM_Process_Diagram.png/598px-CRISP-DM_Process_Diagram.png "CRISP-DM Image from Wikipedia")

You can follow the detailed analysis in the [Jupyter notebook](https://github.com/cdungca/comparing-classifiers/blob/main/prompt_III.ipynb).

## Objective

The objective of this study is to compare the performance of the different classifiers (logistic regression, k-nearest neighbors, decision trees, and support vector machines) and identify the best one suited for predicting the postive outcome of a marketing call.   

## Data Analysis

The data was collected from 17 marketing campaigns in May 2008 until November 2010. Here are the fields included in bank-additional-full.csv:

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

The target is a binary field y which specifies if the marketing contact is successful with the value of "yes."

To start the analysis, we've looked at the data distribution on the categorical fields and performed additional pre-processing. 

Data distributions of some fields:

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/age_distribution.png "Age Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/campaign_distribution.png "Campaign Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/contact_distribution.png "Contact Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/default_distribution.png "Default Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/housing_distribution.png "Housing Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/loan_distribution.png "Loan Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/marital_distribution.png "Marital Distribution")

![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/poutcome_distribution.png "Poutcome Distribution")

Looking at the distribution and the values for each categorical field, there are some records with "unknown" values in job, marital, education, default, loan. We will remove these records to clean the data.

In pdays numeric field, client not previously contacted contains a value of 999. This will affect the scaling of data and we will just replace it with 0. 

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
- duration
- pdays
- previous
- emp.var.rate
- cons.proce.idx
- cons.conf.idx
- euribor3m

Finally, we've split the data between train and test sets.

## Modeling

We need a baseline model to compare the performance of the different classifiers. The baseline model was created using DummyClassifier and here are the accuracy, precision, confusion matrix, and ROC curve.

- Test Accuracy = 0.780635
- Test Precision = 0.130617

Baseline
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/baseline-cmroc.png "Baseline - Confusion Matrix and ROC Curve")

Next, we've created the models and used evaluation metrics such as Train Accuracy, Test Accuracy, Test Precision, and Fit time. Here are the differenct classifiers using the defaul parameters:

Logistic Regression (default parameters)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/lgr_default-cmroc.png "Logistic Regression - Confusion Matrix and ROC Curve")

KNN (default parameters)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/knn_default-cmroc.png "KNN - Confusion Matrix and ROC Curve")

Decision Tree (default parameters)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/dtree_default-cmroc.png "Decision Tree - Confusion Matrix and ROC Curve")

SVM (default parameters)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/svm_default-cmroc.png "SVM - Confusion Matrix and ROC Curve")

The table below shows the perfromace of the different classifiers side by side: 

|Model|Train Accuracy|Test Accuracy|Test Precision|Fit Time (sec)|
|--|--|--|--|--|
|Baseline||0.780635|0.130617||
|Logistic Regression|0.900813|0.901863|0.669267|0.835536|
|KNN|0.922243|0.890186|0.592754|0.751936|
|Decision Tree|1.0|0.872868|0.497930|0.942094|
|SVM|0.914239|0.902125|**0.679803**|32.002124|

As we can see, all 4 classifiers performed better than the baseline model. We've also observed the following:

- SVM is the slowest to train and it took around 32 secs to complete.
- Both Logistic Regression and SVM have 90% accuracy on the test set. 
- SVM has the highest precision which is around 68% (0.6798).
- The training accuracy in the Decision Tree is 1 which means there is overfitting. By default, max_depth is set to none and the nodes will be expanded until all leaves are pure. Adding a limit to the max_depth can avoid overfitting.

Based on the business objective, we would like to predict clients who would say yes to a marketing call. We want the true positive to increase and the false negative to go down. Precision is the indicator that is important to our use case. SVM would give us the highest precision using the table above.

The next step is to tune the paramaters using GridSearchCV. After tuning, we can compare the result with those using the default parameters. Here are the result with tuning:

Logistic Regression (C = 0.01, Solver = liblinear)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/lgr_best-cmroc.png "Logistic Regression - Confusion Matrix and ROC Curve")

KNN (n_neighbors = 3, weights = uniform)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/knn_best-cmroc.png "KNN - Confusion Matrix and ROC Curve")

Decision Tree (max_depth = 5, min_samples_leaf = 2, criterion = gini)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/dtree_best-cmroc.png "Decision Tree - Confusion Matrix and ROC Curve")

SVM (c = 10, gamma = auto)
![alt text](https://github.com/cdungca/comparing-classifiers/blob/main/images/svm_best-cmroc.png "SVM - Confusion Matrix and ROC Curve")

Just like in the previous step, we've included a table with the different metrics:

|Model|Train Accuracy|Test Accuracy|Test Precision|
|--|--|--|--|
|Logistic Regression|0.900726|0.901863|0.668217|
|KNN|0.912271|0.893991|0.620584|
|Decision Tree|0.894385|0.890449|0.598187|
|SVM|0.916601|0.903306|**0.679245**|

SVM is still on top if we look at precision so we will be using it for the predictive model. 

## Next Steps

To further increase precision in SVM, we can try the following:

- Decrease C thereby increasing the strength of the regularization.
- In the data set, there are more records where y="no" then the positive, y="yes". Due to this imbalance, we can try changing class_weights and check precision.




[^1]: Shearer C., The CRISP-DM model: the new blueprint for data mining, J Data Warehousing (2000); 5:13—22.






