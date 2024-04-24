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

## Approach
1. **Data Collection**: Gathered the relevant data which includes various features such as tumor size, shape, texture and margins from Kaggle.
   
2. **Exploratory Data Analysis**: Conducted exploratory analysis to gain insights into the distribution and relationship among the features and the response variable. This step includes data visualisation and statistical analysis to identify important patterns and correlations.
   
3. **Feature Selection**: Employed various techniques such as correlation analysis, z-scores and feature scores to select the most relevant features for inclusion in the predictive models
   
4. **Model Development**: Utilised various machine learning algorithms to build predictive models. We also employed different techniques such as cross validation and hyperparameter tuning to optimise model performance
   
5. **Optimization of Feature Number**: Explored the impact of varying the number of features on model performance to identify the optimal subset that maximises predictive accuracy while minimising complexities
   
6. **Evaluation**: Evaluated the performance of the developed models using accuracy and false negative rates. False negatives occur when the classifier incorrectly predicts a tumor as benign (negative) when it is actually malignant. In our case, false negative rates will be more costly as patients with undetected malignant tumors may not receive timely medical intervention and treatment, leading to delays in treatment.


## Models Used
  - Logistic Regression
  - Decision Tree
  - Random Forest Classifier
  - K-Nearest Neighbours

## Conclusion
Upon extensive training of various models with different subsets of features, we have determined that the `Random Forest Classifier` with the following set of relevant features for predicting breast cancer malignancy:

- concave points_worst

- perimeter_worst

- radius_worst

- concave points_mean

- perimeter_mean

- radius_mean

- area_worst

These features are critical measurements related to tumor textures, characteristics, and size. By focusing on these key attributes, our predictive model can effectively differentiate between malignant and benign tumors, thereby aiding in early detection and treatment planning.

## What did we learn from the project?
- Outliers Detection using `IQR` and `Z-score thresholding`.
- Considerations for Data Balancing.
- Feature selection:
- Making use of `selectKBest` and `r_regression` from sklearn to return feature scores to understand the importance of each features with respect to the outcome.
- After knowing which features are important, using of `Recursive Feature Elimination with Cross Validation` (RFECV) to find **how many** features should we use to best optimise the models
- `Logistic Regression model` - sklearn
- `Random Forest classifier` - sklearn
- `K-Nearest Neighbours classifier` and the iterative method to tune its key parameter- sklearn
- Using `GridSearchCV` to do hyperparameter tuning
- Considered other model evaluation metrics apart from accuracy like `f1`, `precision` and `balanced accuracy`

## Presentation Video 
- https://youtu.be/6WyTn8yf3gs

## References
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data

https://www.singaporecancersociety.org.sg/learn-about-cancer/cancer-basics/common-types-of-cancer-in-singapore.html

https://www.nrdo.gov.sg/docs/librariesprovider3/default-document-library/scr-ar-2021-infographicfaa60392d61d475aaf7a7d71fc928b87.pdf?sfvrsn=51bb444f_0

Various documentations from [sklearn](https://scikit-learn.org/stable)



