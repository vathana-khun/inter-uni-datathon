import pandas as pd
import numpy as np
from xgboost import XGRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('climate_visitor.csv')

print("Dataset shape:", df.shape)
print("Training XGBoost model for 2026 predictions...")

# Handle missing values - fill with median for numerical columns
numerical_cols = ['Visitors', 'MaxTemp', 'MinTemp', 'Rainfall']
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Feature Engineering
# Encode resort names
le_resort = LabelEncoder()
df['Resort_encoded'] = le_resort.fit_transform(df['Resort'])

# Create additional features
df['TempRange'] = df['MaxTemp'] - df['MinTemp']
df['AvgTemp'] = (df['MaxTemp'] + df['MinTemp']) / 2

# Create seasonal features
df['Season'] = df['Week'].apply(lambda x: 
    'Winter' if x in [6, 7, 8, 9] else
    'Spring' if x in [10, 11, 12] else
    'Summer' if x in [13, 14, 15, 1] else
    'Autumn'  # weeks 2, 3, 4, 5
)
le_season = LabelEncoder()
df['Season_encoded'] = le_season.fit_transform(df['Season'])

# Create lag features (previous year's data)
df_sorted = df.sort_values(['Resort', 'Year', 'Week'])
df_sorted['Visitors_lag1'] = df_sorted.groupby(['Resort', 'Week'])['Visitors'].shift(1)
df_sorted['MaxTemp_lag1'] = df_sorted.groupby(['Resort', 'Week'])['MaxTemp'].shift(1)
df_sorted['MinTemp_lag1'] = df_sorted.groupby(['Resort', 'Week'])['MinTemp'].shift(1)
df_sorted['Rainfall_lag1'] = df_sorted.groupby(['Resort', 'Week'])['Rainfall'].shift(1)

# Fill lag features with median values for the first year
for col in ['Visitors_lag1', 'MaxTemp_lag1', 'MinTemp_lag1', 'Rainfall_lag1']:
    df_sorted[col].fillna(df_sorted[col].median(), inplace=True)

# Prepare features for modeling
feature_columns = [
    'Year', 'Week', 'Resort_encoded', 'MaxTemp', 'MinTemp', 'Rainfall',
    'TempRange', 'AvgTemp', 'Season_encoded',
    'Visitors_lag1', 'MaxTemp_lag1', 'MinTemp_lag1', 'Rainfall_lag1'
]

X = df_sorted[feature_columns]
y = df_sorted['Visitors']

# Split the data (use 2020-2024 as test set to validate model performance)
train_mask = df_sorted['Year'] < 2020
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[~train_mask]
y_test = y[~train_mask]

# Train XGBoost model
xgb_model = XGRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50
)

# Fit the model
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# Evaluate model performance
y_pred = xgb_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# Create 2026 prediction data
print("Generating 2026 predictions...")

# Get unique resorts and weeks
resorts = df['Resort'].unique()
weeks = range(1, 16)  # weeks 1-15

# Generate predictions for 2026
results_2026 = []

for resort in resorts:
    resort_data_2024 = df_sorted[(df_sorted['Resort'] == resort) & 
                                 (df_sorted['Year'] == 2024) & 
                                 (df_sorted['Week'].isin(weeks))]
    
    for week in weeks:
        week_data_2024 = resort_data_2024[resort_data_2024['Week'] == week]
        
        if len(week_data_2024) > 0:
            # Use 2024 weather data as baseline for 2026
            row_2024 = week_data_2024.iloc[0]
            
            # Create prediction features
            pred_features = pd.DataFrame([{
                'Year': 2026,
                'Week': week,
                'Resort_encoded': row_2024['Resort_encoded'],
                'MaxTemp': row_2024['MaxTemp'],
                'MinTemp': row_2024['MinTemp'],
                'Rainfall': row_2024['Rainfall'],
                'TempRange': row_2024['TempRange'],
                'AvgTemp': row_2024['AvgTemp'],
                'Season_encoded': row_2024['Season_encoded'],
                'Visitors_lag1': row_2024['Visitors'],
                'MaxTemp_lag1': row_2024['MaxTemp'],
                'MinTemp_lag1': row_2024['MinTemp'],
                'Rainfall_lag1': row_2024['Rainfall']
            }])
            
            # Make prediction
            pred_visitors = xgb_model.predict(pred_features[feature_columns])[0]
            pred_visitors = max(0, int(round(pred_visitors)))  # Ensure non-negative integer
            
            results_2026.append({
                'Year': 2026,
                'Week': week,
                'Resort': resort,
                'Predicted_Visitors': pred_visitors
            })

# Convert to DataFrame and save
results_df = pd.DataFrame(results_2026)
results_df = results_df.sort_values(['Resort', 'Week']).reset_index(drop=True)

# Save to CSV
results_df.to_csv('predicted_numvisitor.csv', index=False)

print(f"Predictions saved to 'predicted_numvisitor.csv'")
print(f"Total predictions generated: {len(results_df)} rows")
print(f"Resorts: {len(resorts)}, Weeks: 1-15")
print("\nPrediction complete!")