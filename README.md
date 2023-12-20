import numpy as np
from scipy.integrate import solve_ivp
import pysindy as ps

# Lorenz system parameters
sigma = 10
beta = 8 / 3
rho = 28

# Lorenz system
def lorenz_system(t, y):
    return [
        sigma * (y[1] - y[0]),
        y[0] * (rho - y[2]) - y[1],
        y[0] * y[1] - beta * y[2]
    ]

# Generate synthetic data
t_span = np.linspace(0, 2, 1000)  # time span
y0 = [-8, 8, 27]  # initial condition
solution = solve_ivp(lorenz_system, [t_span[0], t_span[-1]], y0, t_eval=t_span)

# Initialize and fit the SINDy model
model = ps.SINDy()
model.fit(solution.y.T, t=solution.t)

# Print the learned model
model.print()

# Test the model
# Predict derivatives
y_dot_test = model.differentiate(solution.y.T, t=solution.t)

# Compare y_dot_test with true derivatives
true_derivatives = np.array([lorenz_system(0, y) for y in solution.y.T])
comparison = np.mean(np.abs(y_dot_test - true_derivatives), axis=0)
