import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_forestfire_data(forest_path, ambee_path):
    # Load datasets
    forest_df = pd.read_csv(forest_path)
    ambee_df = pd.read_csv(ambee_path)
    
    # Combine relevant features
    forest_df['temperature'] = forest_df['Temperature']
    forest_df['wind_speed'] = forest_df['Ws']
    forest_df['humidity'] = forest_df['RH']
    
    # Create fire intensity column
    conditions = [
        (forest_df['Classes'] == 'not fire'),
        (forest_df['FWI'] <= 2),
        (forest_df['FWI'] <= 5),
        (forest_df['FWI'] > 5)
    ]
    values = ['No Fire', 'Low', 'Medium', 'High']
    forest_df['fire_intensity'] = np.select(conditions, values, default='Low')
    
    # Create binary target
    forest_df['fire_occurrence'] = (forest_df['Classes'] != 'not fire').astype(int)
    
    # Handle categorical variables
    le = LabelEncoder()
    forest_df['month'] = le.fit_transform(forest_df['month'])
    forest_df['day'] = le.fit_transform(forest_df['day'])
    
    # Select features for modeling
    features = ['temperature', 'wind_speed', 'humidity', 'month', 'day', 
               'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
    
    X = forest_df[features]
    y = forest_df['fire_occurrence']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    return X_scaled, y, forest_df
