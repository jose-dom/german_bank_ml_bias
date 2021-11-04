# German Bank Paper

### Introduction

---

Machine learning is being an ever increasing part of how decisions are made. Machine learning models provide good insight in identifying patterns within a given dataset, however, not all models are development equally. Depending on how a particular model performs  with each application, the risk of algorithmic bias varies from harmless, inconvenience, to life threatening, and criminal. Since algorithms are the products developed to learn and predict an output, it is the data scientist that is able to understanding where unwanted bias appears and remove it anywhere in the process.

The discussion of bias concerns the concept of fairness, which in statistics and machine learning doesn't have one direct meaning. Considering the importance of measuring bias, we will quantify the the amount of bias by producing a table to describe the accuracy of the model.    

### Background

We are using the German Credit Dataset, which contains records on individuals applying for loans with a particular Bank. The features include:

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 21 columns):
 #   Column                   Non-Null Count  Dtype 
---  ------                   --------------  ----- 
 0   status                   1000 non-null   object
 1   duration                 1000 non-null   int64 
 2   credit_history           1000 non-null   object
 3   purpose                  1000 non-null   object
 4   amount                   1000 non-null   int64 
 5   savings                  1000 non-null   object
 6   employment_duration      1000 non-null   object
 7   installment_rate         1000 non-null   object
 8   personal_status_sex      1000 non-null   object
 9   other_debtors            1000 non-null   object
 10  present_residence        1000 non-null   object
 11  property                 1000 non-null   object
 12  age                      1000 non-null   int64 
 13  other_installment_plans  1000 non-null   object
 14  housing                  1000 non-null   object
 15  number_credits           1000 non-null   object
 16  job                      1000 non-null   object
 17  people_liable            1000 non-null   object
 18  telephone                1000 non-null   object
 19  foreign_worker           1000 non-null   object
 20  credit_risk              1000 non-null   object
dtypes: int64(3), object(18)
memory usage: 164.2+ KB
```

We focused on creating a classifier that will be able to determine the overall credit risk outcome of an individual. We are particularly interested in viewing the difference between men and women, as we see initially  that gender is coupled with marital status with the personal_status_sex feature. We will mitigates by analyzing a dataset that separates gender and marital status. 

### Mitigation

---

From looking at our categorical values we can see something different with the personal_status_sex feature. This feature is a combination of the individuals sex and their martial status. This can definitely can a cause for concern when regarding bias. According to the data dictionary:

A91 : male : divorced/separated 

A92 : female : divorced/separated/married 

A93 : male : single A94 : male : married/widowed 

A95 : female : single

We can see that from the values the German Bank is coupling divorced and separated men with value A91. The bank is classifying those two types of men to be the same when qualifying them for credit. This is an assumption the bank has made and given the values of A92, we can see that they view women the same no matter if they are divorced, separated, or currently married. This assumption raises a red flag because classifying these women as the same can create bias when qualifying for credit.

We can try to mitigate against what the bank has done with this particular feature, however, we don't have enough information to decouple this issue completely. With that we can assess that the bank is biased when assessing the credit risk of a customer or group. Clearly the bank has manipulated the data to group by these categories. We will see later on the implications on this on predicting credit_risk. Where we will assess group fairness and measure whether the protected group (males) and unprotected group (females) have equal probability of being assigned to the positive predicted class.

### Statistical Measures & Results

---

Our statistical measures of fairness are explained using a confusion matrix or a table that  describes the accuracy of the classification model. Each model is a logistic regression model trained using 90% of our dataset and tested on the remaining 10%. 

**Model 1 - without mitigation**

The first model considered the following features as predictors for an applicant's overall credit risk outcome.

**Predictors**

- status
- duration
- credit_history
- purpose
- savings
- employment_duration
- installment_rate
- personal_status_sex
- other_debtors
- property
- other_installment_plans
- housing
- number_credits

**Encoders**

personal_status_sex 

```
{'female : non-single or male : single': 0,
 'male : married/widowed': 1,
 'female : single': 2,
 'male : divorced/separated': 3}
```

**Target** - credit_risk

**Model Intercept -**  `-0.44280023`

**Model Coefficients -**

```
-0.66450738
+0.03234895
+0.37310719
-0.03922051
-0.28247456
-0.14242697
-0.17309412
-0.09662876
-0.20090421
+0.14401903
+0.11805857
-0.18844549
+0.00364749
```

**Model Score** - `0.7522222222222222`

**Confusion Matrix** - 

```
[569,  64],
[159, 108]
```

![Untitled](German%20Bank%20Paper%2094445be799084710bea65bf1aa15a9b3/Untitled.png)

```
              precision    recall  f1-score   support

           0       0.78      0.90      0.84       633
           1       0.63      0.40      0.49       267

    accuracy                           0.75       900
   macro avg       0.70      0.65      0.66       900
weighted avg       0.74      0.75      0.73       900
```

1. **True Positive (TP)** = **108**
2. **False Positive (FP)** = **64**
3. **False Negative (FN)** = **159**
4. **True Negative (TN)** = **569**
5. **Positive Predictive Value (PPV)** = $\frac{TP}{TP+FP} = \frac{108}{108+64}$= **0.63529412**
6. **False discovery rate (FDR)** = $\frac{FP}{TP+ FP}$= $\frac{64}{108+64}$= **0.37209302**
7. **False omission rate (FOR)** = $\frac{FN}{TN+FN}$= $\frac{159}{569+159}$= **0.21840659**
8. **Negative predictive value (NPV)** = $\frac{TN}{TN+FN}$= $\frac{569}{569+159}$= **0.78159341**
9. **True positive rate (TPR)** = $\frac{TP}{TP+FN}$ = $\frac{108}{108+159}$= **0.40449438**
10. **False positive rate (FPR)** = $\frac{FP}{FP+TN}$=$\frac{64}{64+569}$= **0.10.110585**
11. **False negative rate (FNR)** = $\frac{FN}{TP+FN}$= $\frac{159}{108+159}$= **0.59550562**
12. **True negative rate (TNR)** = $\frac{TN}{FP+TN}$ = $\frac{569}{569+64}$= **0.89889415**

**Model 2 - with mitigation**

The second model considered the following features as predictors for an applicant's overall credit risk outcome.

**Predictors**

- lbl_checkacct_status
- duration_loanterm
- lbl_credhistory
- lbl_loan_purpose
- lbl_saving_acct_bonds
- lbl_employmt_tenure
- lbl_oth_install_plans
- lbl_other_debt
- lbl_property_type_assets
- lbl_housing_own_rent
- num_credits_atbank
- marital_status
- gender

**Encoders**

gender - {0 - male, 1 - female}

marital_status - {0 - single, 1 - married/widowed/divorced}

**Target** - credit_risk_outcome

**Model Intercept -**  `-1.13319917`

**Model Coefficients -**

```
-0.20949238
+0.03955574
+0.1442243
-0.07166137
-0.27643969
-0.03387984
-0.29061632
-0.09376148
+0.07668025
+0.08027129
-0.15209262
+0.23808244
-0.22277983
```

**Model Score** - `0.7188888888888889`

**Confusion Matrix -** 

```
[590,  35],
[218,  57]
```

![Untitled](German%20Bank%20Paper%2094445be799084710bea65bf1aa15a9b3/Untitled%201.png)

```
              precision    recall  f1-score   support

           1       0.73      0.94      0.82       625
           2       0.62      0.21      0.31       275

    accuracy                           0.72       900
   macro avg       0.67      0.58      0.57       900
weighted avg       0.70      0.72      0.67       900
```

1. **True Positive (TP)** = **57**
2. **False Positive (FP)** = **35**
3. **False Negative (FN)** = **218**
4. **True Negative (TN)** = **590**
5. **Positive Predictive Value (PPV)** = $\frac{TP}{TP+FP} = \frac{57}{57+35}$= **0.61956**
6. **False discovery rate (FDR)** = $\frac{FP}{TP+ FP}$= $\frac{35}{57+35}$= **0.38043478**
7. **False omission rate (FOR)** = $\frac{FN}{TN+FN}$= $\frac{218}{590+218}$= **0.26980198**
8. **Negative predictive value (NPV)** = $\frac{TN}{TN+FN}$= $\frac{590}{590+218}$= **0.73019802**
9. **True positive rate (TPR)** = $\frac{TP}{TP+FN}$ = $\frac{57}{57+218}$= **0.20727273**
10. **False positive rate (FPR)** = $\frac{FP}{FP+TN}$=$\frac{35}{35+590}$= **0.056**
11. **False negative rate (FNR)** = $\frac{FN}{TP+FN}$= $\frac{218}{57+218}$= **0.79272727**
12. **True negative rate (TNR)** = $\frac{TN}{FP+TN}$ = $\frac{590}{35+590}$= **0.944**

### Fairness

Overall, we can determine that the classifier does not result in equal probabilities for both protected and unprotected groups. The data contains Aggregation bias, which is due to the personal_status_sex feature, which is drawing conclusions about individuals by observing their population. The features which were combined were not carefully considered, as the individuals across genders and marital status types are complex and behave differently. This type of general assumption made by coupling samples results in the aggregate bias.

Through the method of our analysis we attempted to reduce the likeliness of algorithmic bias. Being able to use the train_test_split allowed us to create a random sample as to not train and test on non-randomized samples. However, mitigation of the combined feature allowed us to remove some of the aggregate bias present in the data. This was a decision that allowed us to see the difference in predictions between each model. However, it is algorithmic bias since we made this choice which did involve the outcome of our model. 

According to "A Survey on Bias and Fairness in Machine Learning" by NINAREH MEHRABI, FRED MORSTATTER, NRIPSUTA SAXENA, KRISTINA LERMAN, and ARAM GALSTYAN, "Broadly, fairness is the absence of any prejudice or favoritism towards an individual or a group based on their intrinsic or acquired traits in the context of decision-making." With the evaluation considered by these models, it is shown that the predictions are not fair. We had to mitigate the issue during pre-processing and then provide the new data to our classifier. 

### Conclusion

In this analysis, we looked at whether the gender credit dataset was fair and/or biased, when determining credit risk. We were able to demonstrate for the unprotected group (females), they were less likely to be predicted in the positive class. We were able to demonstrate this by comparing two logistic regression classifiers and the statistical metrics associated with each. We then assessed the fairness by referencing Mehrabi and evaluating whether this method was fair; also included was the types of bias.