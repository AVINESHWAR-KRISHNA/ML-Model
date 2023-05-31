import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
import warnings
from sklearn.exceptions import ConvergenceWarning
from keras.callbacks import EarlyStopping
import Feature_Selection_Module
import os
import joblib

# Split the data into input features (X) and target variable (y)
X = Feature_Selection_Module.X
y = Feature_Selection_Module.y
best_params = Feature_Selection_Module.best_params

# Perform one-hot encoding for categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
ct = ColumnTransformer([('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X_encoded = ct.fit_transform(X)

# Apply non-negative scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_scaled)

# Perform feature selection using SelectKBest and f_classif
selector = SelectKBest(f_classif, k='all')
X_selected = selector.fit_transform(X_imputed, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Scale the selected features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Encode the target variable with numeric values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Define the models with the best parameters
models = [
    ('Random Forest', RandomForestClassifier(n_estimators=best_params['n_estimators'],
                                             min_samples_split=best_params['min_samples_split'],
                                             min_samples_leaf=best_params['min_samples_leaf'],
                                             max_depth=best_params['max_depth'])),
    ('Support Vector Machines', SVC(probability=True,
                                    C=1.0, kernel='rbf',
                                    gamma='scale')),
    ('Gradient Boosting', GradientBoostingClassifier(n_estimators=best_params['n_estimators'],
                                                     learning_rate=0.1,
                                                     max_depth=5,
                                                     subsample=1.0,
                                                     min_samples_split=best_params['min_samples_split'],
                                                     min_samples_leaf=best_params['min_samples_leaf'])),
    ('XGBoost', XGBClassifier(learning_rate=0.1,
                              max_depth=5,
                              n_estimators=best_params['n_estimators'],
                              subsample=1.0)),
]

# Ignore convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Specify the folder path where the saved model files are located
saved_models_folder = "/Users/avineshwarkrishnasingh/Desktop/Trained_Models"

# Define a dictionary to store the trained models
trained_models = {}

# Check if the saved models folder exists
if os.path.exists(saved_models_folder):
    # Iterate over the files in the folder
    for filename in os.listdir(saved_models_folder):
        # Check if the file is a pickle file
        if filename.endswith(".pkl"):
            # Load the model from the file
            model_name = os.path.splitext(filename)[0]  # Extract model name from file name
            model_path = os.path.join(saved_models_folder, filename)  # Build model file path
            loaded_model = joblib.load(model_path)
            trained_models[model_name] = loaded_model

# Perform model training and evaluation
results = {}
print("--------------------------------------------------------")
for model_name, model in models:
    print(f"Training {model_name}...")
    if model_name in trained_models:
        # If the model has already been trained and loaded, use the loaded model
        print(f"Reusing existing model: {model_name}")
        loaded_model = trained_models[model_name]
        test_score = loaded_model.score(X_test, y_test)
    else:
        if isinstance(model, Sequential):  # Apply early stopping for sequential models
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            history = model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, batch_size=32, epochs=20,
                                validation_split=0.2, callbacks=[early_stopping], verbose=0)
            test_score = model.evaluate(X_test.reshape(X_test.shape[0], X_test.shape[1], 1), y_test, verbose=0)[1]
        else:
            model.fit(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
        # Save the trained model for future use
        model_path = os.path.join(saved_models_folder, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        trained_models[model_name] = model
    results[model_name] = {'test_score': test_score}
print("--------------------------------------------------------\nAll models trained successfully.\n--------------------------------------------------------\n")

# Display results
for model_name, result in results.items():
    print(f"Model Name: {model_name}")
    print(f"Test Score: {round(result['test_score'], 2)}")
