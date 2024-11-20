import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import joblib

# Load data
data = pd.read_csv('student-mat.csv')

# Preprocess data, encoding categorical data into numerical labels using LabelEncoder.
le = LabelEncoder()
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])
# drop the school and grade columns
data = data.drop(['school', 'G1', 'G2'], axis='columns')
# Find correlations with the Grade
most_correlated = data.corr().abs()['G3'].sort_values(ascending=False)
# Maintain the top 8 most correlation features with Grade
most_correlated = most_correlated[:9]
data = data.loc[:, most_correlated.index]
df = pd.DataFrame(data)
new_order = ['age', 'Fedu', 'Medu', 'reason', 'higher', 'goout', 'romantic', 'failures', 'G3']
df = df[new_order]
# Save the preprocessed data to a new CSV file
df.to_csv('preprocessed_data.csv', index=False)
# Split data into features and target
X = data.drop(columns=['G3'])
y = data['G3']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Train LinearRegression model
from sklearn.linear_model import LinearRegression
LReg_model = LinearRegression()
LReg_model= LReg_model.fit(X_train, y_train)
# Predict on the test set
y_pred = LReg_model.predict(X_test)
# Save the model
joblib.dump(LReg_model, 'LReg_model1.joblib')

'''
# Train KNN model
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
# Predict on the test set
y_pred = knn_model.predict(X_test)
# Save the model
joblib.dump(knn_model, 'KNN_model1.joblib')
'''
'''
# Train Naive Bayes model
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)
# Predict on the test set
y_pred = nb_model.predict(X_test)
# Save the model
joblib.dump(nb_model, 'nb_model1.joblib')
'''
'''
# Train Log Regression medel
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model = lg_model.fit(X_train, y_train)
# Predict on the test set
y_pred = lg_model.predict(X_test)
# Save the model
joblib.dump(lg_model, 'lg_model1.joblib')
'''
'''
# Train SVM model
from sklearn.svm import SVC
svm_model = SVC()
svm_model = svm_model.fit(X_train, y_train)
# Predict on the test set
y_pred = svm_model.predict(X_test)
# Save the model
joblib.dump(svm_model, 'svm_model1.joblib')
'''
# Compute Root Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mabr = mean_absolute_error(y_test, y_pred)
r_2_s=r2_score(y_test, y_pred)

joblib.dump(rmse, 'rmse_1.joblib')
joblib.dump(mabr, 'MASR.joblib')
joblib.dump(r_2_s, 'MASR1.joblib')




