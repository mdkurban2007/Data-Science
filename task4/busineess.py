import pulp

# --- STEP 1: PROBLEM SETUP ---
# Defining the business problem as a maximization problem
# We want to maximize total profit from production [cite: 76, 77]
model = pulp.LpProblem("Business_Profit_Optimization", pulp.LpMaximize)

# --- STEP 2: DECISION VARIABLES ---
# X = Units of Product A to produce
# Y = Units of Product B to produce
# Products cannot be negative, and we produce whole units (Integer)
X = pulp.LpVariable('Product_A_Units', lowBound=0, cat='Integer')
Y = pulp.LpVariable('Product_B_Units', lowBound=0, cat='Integer')

# --- STEP 3: OBJECTIVE FUNCTION ---
# Objective: Maximize Profit
# Let's say Profit for A = $50 and Profit for B = $80
model += 50 * X + 80 * Y, "Total_Profit"

# --- STEP 4: CONSTRAINTS ---
# Every business has limits (Resource Constraints) 

# Constraint 1: Labor Hours (A takes 2 hrs, B takes 3 hrs. Total available = 100 hrs)
model += 2 * X + 3 * Y <= 100, "Labor_Limit"

# Constraint 2: Raw Material (A takes 1 unit, B takes 4 units. Total available = 120 units)
model += 1 * X + 4 * Y <= 120, "Material_Limit"

# --- STEP 5: SOLVING THE PROBLEM ---
# Using the PuLP solver to find the optimal solution 
model.solve()

# --- STEP 6: INSIGHTS & RESULTS ---
# Displaying the results as required by the task deliverable 
print(f"Optimization Status: {pulp.LpStatus[model.status]}")
print(f"Optimal quantity of Product A: {X.varValue}")
print(f"Optimal quantity of Product B: {Y.varValue}")
print(f"Maximum Business Profit: ₹{pulp.value(model.objective)}") 

# --- STEP 7: BUSINESS INSIGHTS ---
# Based on the results, the company should focus on the above quantities 
# to ensure resources are used efficiently while maximizing revenue.