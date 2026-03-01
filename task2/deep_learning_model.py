import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Real Dataset from GitHub
def load_data():
    # GitHub se Titanic dataset ka raw link
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Cleaning: Hum sirf numerical columns le rahe hain simple rakhne ke liye
    # Survived hamara 'Target' (Label) hai
    df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].dropna()
    
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Split data (80% Training, 20% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling: Deep Learning ke liye zaroori hai
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# 2. Build Neural Network Model
def build_model(input_shape):
    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_shape,)), # Pehli layer
        Dense(8, activation='relu'),                              # Doosri layer
        Dense(1, activation='sigmoid')                            # Output (Survive: Yes/No)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("--- Task 2: Deep Learning with GitHub Dataset ---")
    
    # Data load karein
    X_train, X_test, y_train, y_test = load_data()
    
    # Model banayein
    model = build_model(X_train.shape[1])
    
    # Training
    print("\nTraining on Titanic Dataset...")
    model.fit(X_train, y_train, epochs=30, batch_size=10, verbose=1)
    
    # Accuracy check karein
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nFinal Model Accuracy: {accuracy * 100:.2f}%")
    
    # Save karein
    model.save('titanic_model.h5')
    print("\nModel saved successfully as 'titanic_model.h5'")