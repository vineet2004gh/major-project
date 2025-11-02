import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- 1. Load Data ---
df = pd.read_csv('dataset/processed_dataset/training_ndvi_data_2.csv')
print("Successfully loaded 'training_ndvi_data_2.csv'")
# --- 2. Define Features (X) and Target (y) ---
df = df.dropna()
X = df.drop('crop_yield', axis=1)
y = df['crop_yield']

# --- 3. Define Preprocessing Steps ---
# We need to scale numbers and encode text categories.

# Identify categorical and numerical feature names
numeric_features = ['ndvi_max', 'ndvi_mean', 'ndvi_min', 'ndvi_std_dev', 'season_length']
categorical_features = ['crop_name', 'district']

# Create a transformer for numeric features (scaling)
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a transformer for categorical features (one-hot encoding)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine these steps using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 4. Create the Full Model Pipeline ---
# This chains the preprocessing and the model together
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# --- 5. Split Data and Train Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining the model...")
model.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
print("\n--- Model Evaluation ---")
y_pred = model.predict(X_test)

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Show a few predictions
print("\n--- Example Predictions ---")
print(f"Actual values:    {y_test.values[:5]}")
print(f"Predicted values: {[round(p, 2) for p in y_pred[:5]]}")