# CODTECH IT SOLUTIONS - DATA SCIENCE INTERNSHIP

This repository contains the tasks completed during my Data Science internship at CODTECH IT SOLUTIONS. The projects focus on end-to-end machine learning deployment and business process optimization.

---

## Projects Overview

### 1. Task 3: End-to-End Data Science Project (House Price Prediction)
**Objective:** Develop a full data science pipeline from data collection to deployment using Flask.

* **Algorithm:** Linear Regression
* **Tech Stack:** Python, Pandas, Scikit-Learn, Flask, HTML
* **Functionality:** A web application where users can input the square footage of a house and get an estimated price in Lakhs.
* **Key Files:** * `train_model.py`: Script to train and save the model.
    * `app.py`: Flask application to serve the model via a web UI.
    * `house_model.pkl`: The serialized trained model.

### 2. Task 4: Optimization Model (Business Profit Maximization)
**Objective:** Solve a business resource allocation problem using Linear Programming.

* **Library:** PuLP
* **Problem Statement:** Maximizing profit for a manufacturing unit producing two products (A & B) under labor and raw material constraints.
* **Deliverable:** A script/notebook demonstrating the setup, mathematical solution, and business insights.
* **Insights:** The model successfully identifies the optimal production quantity to achieve a maximum profit of ₹2,640.

---

##  How to Run the Projects

### Prerequisites
Ensure you have Python installed, then install the required libraries:
```bash
pip install pandas scikit-learn flask pulp
