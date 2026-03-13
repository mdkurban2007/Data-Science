from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# --- STEP 1: LOADING THE TRAINED MODEL ---
# The model saved by train_model.py is loaded here
try:
    with open('house_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Error: house_model.pkl not found. Run train_model.py first.")

# --- STEP 2: USER INTERFACE (HTML) ---
html_layout = '''
    <div style="text-align: center; margin-top: 50px; font-family: 'Arial', sans-serif;">
        <h2 style="color: #2c3e50;">House Price Prediction System</h2>
        <p>End-to-End Data Science Project (Task 3)</p>
        <form action="/predict" method="post" style="margin-top: 20px;">
            <input type="number" name="area" placeholder="Enter Area (e.g. 1200)" required 
                   style="padding: 10px; width: 200px; border-radius: 4px; border: 1px solid #ccc;">
            <button type="submit" style="padding: 10px 20px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer;">
                Predict Price
            </button>
        </form>
    </div>
'''

@app.route('/')
def index():
    return render_template_string(html_layout)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input from the form
        area_input = float(request.form['area'])
        
        # Making prediction using the loaded model
        prediction = model.predict(np.array([[area_input]]))[0]
        
        return f"""
            <div style="text-align: center; margin-top: 50px; font-family: Arial;">
                <h3 style="color: #27ae60;">Prediction Result</h3>
                <p>For an area of {area_input} sq ft, the estimated price is: <b>₹{round(prediction, 2)} Lakhs</b></p>
                <hr style="width: 40%;">
                <a href="/" style="text-decoration: none; color: #3498db;">← Back to Home</a>
            </div>
        """
    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>Go back</a>"

if __name__ == "__main__":
    # Starting the Flask application [cite: 61]
    app.run(debug=True)