## Credit Card Fraud Project

Data has been previously treated for privacy reasons, therefore the dataset used is a version processed with principal component analysis (PCA), except for the fields of “Time”, “Amount” and “Class”.  This report presents the design and methodology on this project. The structure will be Exploratory Data Analysis next, followed by data cleaning, handling imbalance in the dataset, model selection, testing, and hyperparameter tuning.

### EDA:

Initial check on data shows that the data ranges differ greatly amongst all features. Normalization should be done, mean and median are similar in most of the features which shows symmetrical distribution, and Info shows no null data and same data type across all features except Class. The dataset has two unique classes of type int64, where 0 is the non-fraud class and 1 represents a fraud. Class is correlated moderately with V3, V8, V12 V14 and V17. As the feature names are not present (due to the PCA processing) all the features will be used. 

Using Boxplots in order to visualize outliers and ranges I could find that all variables contain high number of outliers, all variables have a median close to 0, and all columns have different ranges, comparison is better after normalization to determine dispersion. Plotting the distribution shows that the distribution on all the variables appear to be normal with small variations in V1(Positive skewed) and V3 (negative skewed).

### Data Cleaning:

As learnt from the visualization, the dataset has many outliers. In order to identify, the Tukey method was applied to find the percentage of outliers in every column. It ranges from 1,18% to 11,20% amongst all columns with only the time column not having any. For this I replaced the outliers with either the minimum threshold or the maximum, calculated using the Interquartile range and the 25th and 75th quartiles. Then, the columns that weren’t previously scaled were processed with a standard scaler (“Amount” and “Time”). If any NaN was in it was filled with the mean of the column, but no NaN were found.


### Handling imbalance dataset:

The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. 0 Class has 284315 and 1 Class only 492. This imbalance should be addressed to avoid the model generating bias towards the 0 class. It is expected that the model trained on this class scores high as the probability of mislabelling is very low.  I create several datasets using either undersampling or oversampling to test and choose the best for model selection. I use the library imblearn to use the undersampling techniques:  Edited NearestNeighbours and RandomOverSampler and oversampling: Adasyn and RandomOverSampler and SMOTE which is a combination of both. In total I tested 6 datasets. Further tests will show if under-sampling or oversampling works better with the model. Additionally which of each technique is better.


### Model Selection: 

For this section, I choose two models. The first one is a random forest classifier as it has proven to be one of the most accurate for classification. The second one is a neural network with 3 Dense layers and 1 Dense layer with sigmoid for the prediction. It uses Adam as optimizer and binary cross entropy as the loss function and 0.5 dropout.  The accuracy results are presented in the following table for all the six datasets and both models:




|                      | Random Forest           | Neural Network  |
| -------------------- |:-----------------------:| ---------------:|
| Original             | 0.99                     | 0.99 |
| Random Undersampling | 0.92      |   0.93 |
| Edited Undersampling | 0.93      |   0.93 |
|  Random Oversampling |0.99|0.99|
|  Adasyn Oversampling |0.99|0.99|
|  Smote Oversampling |0.99|0.99|

The best performing model for the random forest model uses Random Undersampling dataset. Oversampling seems to create near 100% accuracy but also overfitting. Same as the original dataset. The Neural Network’s best model uses Random Undersampling. Same as with the Random Forest classifier, the use of oversampling obtained 99.8 and above accuracy. It might be a clear case of overfitting. Therefore, choosing the best next option. For both the best option is the model processed with Random Undersampling. This will be the dataset used for further hyperparameter tuning.


### Hyperparameter tuning: 

I performed GridSearch on both models. For the Random Forest I used the parameters: 

```
   "criterion": ("gini", "entropy"),
    "min_samples_leaf": list(range(1,10)),
    "max_depth": list(range(1,10))
```

And as a result the best parameters are: `{'criterion': 'gini', 'max_depth': 6, 'min_samples_leaf': 3}` with an accuracy of 0.922.

For the Neural Network I used: 
```
    "epochs": list(range(10,30)),
    'activation': ['relu', 'tanh'],
    "optimizer": ["Adam", "SDG", 
```

And as a result the best parameters are: `{'activation': 'tanh', 'epochs': 26, 'optimizer': 'Adam'}` with an accuracy of 0.94

One model for each was trained and saved for future inference.


As a conclusion, the results show that the Neural Network has better accuracy (94%)for the same dataset: Random Undersampling which also yields the best results avoiding the overfitting caused by the oversampling. As the dataset is very imbalanced, the Undersampling drops most of the 0 class which is the non-fraud transactions. Future work should focus on either using data augmentation to create more examples of fraudulent transactions to balance the dataset or finding a bigger dataset. 


