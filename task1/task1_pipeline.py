import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# STEP 1: EXTRACT (Loading from Local Host / Computer)
def load_local_data():
    file_path = "stockdata.csv" # Make sure the file is in the same folder
    
    if os.path.exists(file_path):
        print(f"--- Loading data from local file: {file_path} ---")
        df = pd.read_csv(file_path)
        return df
    else:
        print("Error: 'stockdata.csv' not found in the folder!")
        return None

# STEP 2: TRANSFORM (Automated Preprocessing Pipeline)
def transform_data(df):
    # Removing 'Date' column for scaling as it's not a number
    numeric_features = df.columns.drop('Date')
    
    # Building the pipeline: Cleaning + Scaling
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), 
        ('scaler', StandardScaler())                
    ])

    # Applying the transformation logic
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Executing the automated pipeline
    processed_data = preprocessor.fit_transform(df)
    
    # Converting back to a clean DataFrame
    processed_df = pd.DataFrame(processed_data, columns=numeric_features)
    processed_df['Date'] = df['Date'] 
    return processed_df

# STEP 3: LOAD / EXECUTE
if __name__ == "__main__":
    raw_df = load_local_data()
    
    if raw_df is not None:
        final_df = transform_data(raw_df)
        
        print("\n--- Pipeline Execution Successful ---")
        print("Processed Local Stock Data (First 5 rows):")
        print(final_df.head())
        
        # Save the final result
        final_df.to_csv("processed_local_data.csv", index=False)
        print("\n[SUCCESS] Automated ETL process finished. Result saved to 'processed_local_data.csv'.")