import numpy as np

# Input from the user
n = int(input("Enter the number of variables: "))
m = int(input("Enter the number of constraints: "))

C = np.array([float(x) for x in
              input("Enter the coefficients of the objective function separated by spaces: ").split()
              ])
A = np.zeros((m, n))
b = np.zeros(m)

for i in range(m):
    A[i] = np.array([float(x) for x in
                     input(f"Enter the coefficients of constraint {i+1} separated by spaces: ")
                    .split()]
                    )
    b[i] = float(input(f"Enter the right-hand side value of constraint {i+1}: "))

accuracy = float(input("Enter the approximation accuracy: "))
