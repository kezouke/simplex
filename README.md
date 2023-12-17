# Simplex Algorithm for Linear Programming Problems

This repository contains a Python implementation of the Simplex algorithm for solving Linear Programming Problems (LPPs). The Simplex algorithm is an iterative method that optimizes a linear objective function subject to linear equality and inequality constraints.

## Usage

### Prerequisites
- Python 3
- NumPy

### Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/simplex-algorithm.git
   cd simplex-algorithm
   ```

2. Run the provided Python script:
   ```bash
   python simplex_algorithm.py
   ```

3. Follow the on-screen instructions to input the details of your Linear Programming Problem.

4. The script will output the results of the Simplex algorithm, including the optimal solution and objective function value.

### Example

```python
import numpy as np
from simplex_algorithm import simplex

# Define the coefficients of the objective function
obj_function_coefficients_C = np.array([2, -1])

# Define the coefficients of the constraints
non_basic_coefficients_A = np.array([[1, 1], [-1, 2]])

# Define the right-hand side values of the constraints
right_hand_side_vector_b = np.array([4, 3])

# Specify the type of problem (1 for minimization, 2 for maximization)
problem_type = 2

# Run the Simplex algorithm
C_b, Xb, basic_var_nums, non_basic_var_nums, decision_vector = simplex(
    obj_function_coefficients_C, non_basic_coefficients_A, right_hand_side_vector_b, problem_type - 1
)

# Display the results
print(f"Basic variables' coefficients: {C_b}")
print(f"Basic variable numbers: {basic_var_nums}")
print(f"Basic variable values: {Xb}")
print(f"Non-basic variable numbers: {non_basic_var_nums}")
print(f"Decision vector: {decision_vector}")
```

## Function Documentation

The core of the Simplex algorithm is implemented in the `simplex` function. For detailed information about the function parameters and return values, refer to the function documentation provided in the source code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
