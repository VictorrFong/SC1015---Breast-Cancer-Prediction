# Breast-Cancer Prediction Repository
![image](https://github.com/VictorrFong/SC1015---Breast-Cancer-Prediction/assets/162713262/18c0aa7f-9a1c-49e9-a9e1-f93c2a5eeb1d)

## About
This is a mini project that tackles a classification problem. We aim to leverage on different machine learning models to find one which best predict the malignancy of a breast tumor (benign/malignant). 

## Contributors
- @Fong Zheng Feng Victor - EDA, Logistic Regression
- @Thuvaarakesh Kiruparan - EDA, Decision Tree, Random Forest Classifier
- @Tiang Soon Yong - EDA, K-Nearest Neighbours

## Problem Definition
![Screenshot 2024-04-19 103146](https://github.com/VictorrFong/SC1015---Breast-Cancer-Prediction/assets/162713262/e00c0613-4d5c-444b-b103-025a0a9562bb)

As we can see, breast cancer is the leading cause of death for women in Singapore. Early and accurate detection plays a crucial role in improving treatment outcomes and reducing mortality rates. By accurately predicting the malignancy of breast tumors, healthcare providers can intervene promptly and initiate appropriate treatment strategies.

Hence we would like to find an optimal model that accurately predicts the malignancy of a tumor using the relevant features and the optimal **number** of features. 


## Models Used
  - Logistic Regression
  - Decision Tree
  - Random Forest Classifier
  - K-Nearest Neighbours

## What did we learn from the project?
- Outliers Detection using IQR and Z-score thresholding.
- Considerations for Data Balancing.
- Feature selection:
- Making use of selectKBest from sklearn to return feature scores to understand the importance of each features with respect to the outcome.
- After knowing which features are important, using of Recursive Feature Elimination with Cross Validation (RFECV) to find **how many** features should we use to best optimise the models
- Logistic Regression model - sklearn
- Random Forest classifier - sklearn
- K-Nearest Neighbours classifier and the iterative method to tune its key parameter- sklearn
- Using GridSearchCV to do hyperparameter tuning

## References
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

https://www.singaporecancersociety.org.sg/learn-about-cancer/cancer-basics/common-types-of-cancer-in-singapore.html

https://www.nrdo.gov.sg/docs/librariesprovider3/default-document-library/scr-ar-2021-infographicfaa60392d61d475aaf7a7d71fc928b87.pdf?sfvrsn=51bb444f_0

Various documentations from [sklearn](https://scikit-learn.org/stable)



