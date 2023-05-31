import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils import parallel_backend

import Info_Ticker
from Data_Processing_Module import preprocess_data


# Preprocess data
data = preprocess_data(Info_Ticker.symbol, Info_Ticker.period, Info_Ticker.interval).round(2)

# Split the data into input features (X) and target variable (y)
X = data[['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'Stochastic', 'Bollinger_Band', 'MACD']]
y = data['Signal']

# Define preprocessing steps
preprocessing_steps = [
    ('imputer', SimpleImputer()),
    ('scaler', MinMaxScaler()),
]

# Create the preprocessing pipeline
preprocessing_pipeline = Pipeline(preprocessing_steps)

# Apply preprocessing pipeline to input features
X_processed = preprocessing_pipeline.fit_transform(X)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature Importance Evaluation using SelectKBest
k = 5  # Select top 5 features
selector = SelectKBest(score_func=chi2, k=k)
X_new = selector.fit_transform(X_processed, y_encoded)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

# Create the updated dataset with selected features
updated_data = pd.DataFrame(X_new, columns=selected_features)
updated_data['Signal'] = y

# Split the updated data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encoded, test_size=0.2, random_state=42)

# Perform Randomized Search for hyperparameter optimization
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_classifier = RandomForestClassifier(random_state=42)
randomized_search = RandomizedSearchCV(rf_classifier, param_grid, cv=5, n_iter=10, random_state=42, n_jobs=-1)
randomized_search.fit(X_train, y_train)

# Extract the best model and its parameters
best_model = randomized_search.best_estimator_
best_params = randomized_search.best_params_

# Make predictions on the testing set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Best Parameters:", best_params)

# #Exporting the data in csv format.
# data.to_csv('Signal_Generated.csv')
