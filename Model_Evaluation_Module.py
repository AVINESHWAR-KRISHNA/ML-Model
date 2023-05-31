from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import os
import Model_Training_Module

# Assuming you have the predicted labels and true labels from the test dataset
X = Model_Training_Module.X
y = Model_Training_Module.y
best_params = Model_Training_Module.best_params

label_encoder = LabelEncoder()

# Fit and transform the target variable
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Perform feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Define the models with their respective hyperparameters
models = [
    ('Random Forest', RandomForestClassifier()),
    ('Support Vector Machines', SVC(probability=True)),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('XGBoost', XGBClassifier()),
]

# Define the hyperparameter grid for GridSearchCV
param_grid = {
    'Random Forest': {
        'n_estimators': [best_params['n_estimators']],
        'min_samples_split': [best_params['min_samples_split']],
        'min_samples_leaf': [best_params['min_samples_leaf']],
        'max_depth': [best_params['max_depth']]
    },
    'Support Vector Machines': {
        'C': [1.0],
        'kernel': ['rbf']
    },
    'Gradient Boosting': {
        'n_estimators': [best_params['n_estimators']],
        'learning_rate': [0.1],
        'max_depth': [5],
        'subsample': [1.0],
        'min_samples_split': [best_params['min_samples_split']],
        'min_samples_leaf': [best_params['min_samples_leaf']]
    },
    'XGBoost': {
        'n_estimators': [best_params['n_estimators']],
        'learning_rate': [0.1, 0.5, 1.0],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
}

# Function to create the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train_scaled.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Initialize results dictionary
results = {}

# Evaluate each model using cross-validation
for model_name, model in models:
    result = {}

    # Perform hyperparameter tuning using GridSearchCV
    if model_name in param_grid:
        grid_search = GridSearchCV(model, param_grid[model_name], cv=3, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        best_params = grid_search.best_params_
        model.set_params(**best_params)
        result['Best Params'] = best_params

    # Perform cross-validation
    cv_scores = []
    classifier = model
    if model_name == 'LSTM':
        classifier = create_lstm_model()
        cv_scores = cross_val_score(
            classifier,
            X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1),
            y_train,
            cv=3,
            scoring='accuracy'
        )
    else:
        cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=3, scoring='accuracy')

    result['Cross-Validation Scores'] = cv_scores

    # Store the results for the model
    results[model_name] = result

# Evaluate models on the test set
test_scores = {}
for model_name, model in models:
    classifier = model
    if model_name == 'LSTM':
        classifier = create_lstm_model()
        classifier.fit(
            X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1),
            y_train
        )
        y_pred = classifier.predict(X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
    else:
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

    test_scores[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

# Display the results
for model_name, result in results.items():
    print(f"\n{model_name} Results:")
    for metric_name, score in result.items():
        if metric_name == 'Cross-Validation Scores':
            print(f"\n{metric_name}: {score}")
        elif metric_name == 'Best Params':
            print(f"{metric_name}: {score}")
        else:
            print(f"{metric_name}: {score}")

print("\nTest Set Results:")
for model_name, scores in test_scores.items():
    print(f"\n{model_name}:")
    for metric_name, score in scores.items():
        print(f"{metric_name}: {score}")

print("\n---------------------------------------------------------------\nComparing with Existing Models\n---------------------------------------------------------------\n")

# Create the folder if it doesn't exist
folder_path = "Trained_Models"
os.makedirs(folder_path, exist_ok=True)

# Load existing models
existing_models = {}
for model_name, model in models:
    file_path = os.path.join(folder_path, f"{model_name}.pkl")
    if os.path.exists(file_path):
        existing_models[model_name] = joblib.load(file_path)

# Compare the existing models with the newly trained models
models_retrained = False  # Set to True if there are changes in the scores and retraining is needed
for model_name, model in models:
    if model_name in existing_models:
        existing_model = existing_models[model_name]
        new_model = model

        # Compare the models using accuracy score
        existing_score = existing_model.score(X_test_scaled, y_test)
        new_score = test_scores[model_name]['Accuracy']
        print(f"{model_name}: Existing Score - {existing_score:.4f}, New Score - {new_score:.4f}")

        # Compare the scores and decide if retraining is needed
        if abs(existing_score - new_score) >= 0.01:
            models_retrained = True
            existing_models[model_name] = new_model
            joblib.dump(new_model, os.path.join(folder_path, f"{model_name}.pkl"))
            print(f"Changes detected in {model_name}. Retraining and saving the model.")
        else:
            print(f"No significant changes detected in {model_name}. Existing model retained.")
    else:
        models_retrained = True
        existing_models[model_name] = model
        joblib.dump(model, os.path.join(folder_path, f"{model_name}.pkl"))
        print(f"{model_name} saved.")

# Update the existing models if retraining was performed
if models_retrained:
    Model_Training_Module.existing_models = existing_models
    print("\nExisting models updated.")

# You can continue using the existing_models dictionary for future comparisons or operations
