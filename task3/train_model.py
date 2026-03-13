import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np

# --- STEP 1: DATA PREPARATION ---
# Data integrated directly into the script as requested
def prepare_data():
    # Dataset representing Area in sq ft and Price in Lakhs
    data = {
        'Area_sqft': [500, 750, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 3000],
        'Price_Lakhs': [25, 38, 50, 62, 75, 88, 105, 115, 130, 155]
    }
    return pd.DataFrame(data)

# --- STEP 2: MODEL TRAINING ---
def train_and_save_model():
    df = prepare_data()
    
    # Defining independent variable (X) and dependent variable (y)
    X = df[['Area_sqft']]
    y = df['Price_Lakhs']
    
    # Using Linear Regression as the core algorithm [cite: 55]
    model = LinearRegression()
    model.fit(X, y)
    
    # --- STEP 3: EXPORTING THE MODEL ---
    # Saving the model to a file named 'house_model.pkl'
    with open('house_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    
    print("Success: Model trained and saved as 'house_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()