import numpy as np


np.seterr(all="ignore")


def simplex_iteration(A_nb, A_b, C_nb, C_b, b, problem_type, basic_var_nums, non_basic_var_nums):
    """

    :param A_nb: matrix containing non-basic variables coefficients from constraints

    :param A_b: matrix containing basic variables coefficients from constraints
                !also known as matrix B in Advanced Simplex Algorithm (ASA) from lecture 5

    :param C_nb: row vector containing non-basic variables coefficients from objective function

    :param C_b: row vector containing basic variables coefficients from objective function

    :param b: right hand vector b

    :param problem_type:  0/1 = minimization/maximization

    :param basic_var_nums:  list of current basic variables

    :param non_basic_var_nums:  list of current non-basic variables

    :return: optimal solution for LPP
    """
    # find inverse of A_b
    A_b_inv = np.linalg.inv(A_b)
    # find new basic solutions
    Xb = np.matmul(A_b_inv, b)

    # Also known as "Z_j - C_j = C_b_j * B_j_inverse * P_j - C_j" in ASA:
    temp_matrix = np.subtract(C_b @ A_b_inv @ A_nb, C_nb)

    if problem_type == 0:
        # search for entering variable if it is min problem
        enteringVar = np.argmax(temp_matrix)

        # if we can't improve objective function more => return its value
        if temp_matrix[enteringVar] < 0:
            return C_b, Xb, basic_var_nums, non_basic_var_nums
    else:
        # search for entering variable if it is max problem
        enteringVar = np.argmin(temp_matrix)

        # if we can't improve objective function more => return its value
        if temp_matrix[enteringVar] > 0:
            return C_b, Xb, basic_var_nums, non_basic_var_nums

    # update non-basic variables coefficients with respect to current pivot
    newA = np.matmul(A_b_inv, A_nb[:, enteringVar])

    # calculate ratios to determine new exiting (leaving) variable
    ratios = np.divide(Xb, newA)
    exitingVar = np.argmin(ratios[ratios > 0])
    ttrtrtrt = C_b @ Xb
    # swap entering and exiting variables into following matrices
    # var., what was non-basic before comes basic
    # var., what was basic before comes non-basic
    A_nb[:, enteringVar], A_b[:, exitingVar] = A_b[:, exitingVar].copy(), A_nb[:, enteringVar].copy()
    # do the same logic swap for objective function's vectors of coefficients
    C_b[exitingVar], C_nb[enteringVar] = C_nb[enteringVar], C_b[exitingVar]
    # and save the new numbers of basic variables
    basic_var_nums[exitingVar], non_basic_var_nums[enteringVar] = \
        non_basic_var_nums[enteringVar], basic_var_nums[exitingVar]

    # run new iteration
    return simplex_iteration(A_nb, A_b, C_nb, C_b, b, problem_type, basic_var_nums, non_basic_var_nums)


def simplex(C, A, b, problem_type):
    # n - number of variables
    # m - number of constraints
    m, n = A.shape
    non_basic_var_nums = np.arange(1, n + 1)
    basic_var_nums = np.arange(n + 1, n + m + 1)
    decision_vector = np.zeros(n)

    # on the zero iteration basic variable coefficients are unit vectors:
    B = np.identity(m)
    # and zeros in the objective function
    Cb = np.zeros(m)

    C_b, Xb, basic_var_nums, non_basic_var_nums = simplex_iteration(A, B, C, Cb, b,
                                                                    problem_type,
                                                                    basic_var_nums,
                                                                    non_basic_var_nums)

    mask = np.argsort(np.hstack((basic_var_nums, non_basic_var_nums)))
    Xb_extended = np.hstack((Xb, np.zeros(n)))
    decision_vector = Xb_extended[mask][:n]

    return C_b, Xb, basic_var_nums, non_basic_var_nums, decision_vector


# Input from the user

while True:
    print("""
    Choose type of problem:
        1. minimization
        2. maximization
    enter only number '1' or '2': """)
    minmax = int(input().strip())
    if minmax != 1 and minmax != 2:
        print("wrong format was entered!")
        continue

    var_num = int(input("Enter the number of variables: "))
    constr_num = int(input("Enter the number of constraints: "))

    print("Enter the coefficients of the objective function separated by spaces: ")
    obj_function_coefficients_C = np.array([float(x) for x in input().split()])
    non_basic_coefficients_A = np.zeros((constr_num, var_num))
    right_hand_side_vector_b = np.zeros(constr_num)

    for i in range(constr_num):
        print(f"Enter the coefficients of constraint {i + 1} separated by spaces: ")
        non_basic_coefficients_A[i] = np.array([float(x) for x in input().split()])

        print(f"Enter the right-hand side value of constraint {i + 1}: ")
        right_hand_side_vector_b[i] = float(input())

    accuracy = float(input("Enter the approximation accuracy: "))

    try:
        C_b, Xb, basic_var_nums, non_basic_var_nums, decision_vector = simplex(obj_function_coefficients_C,
                                                                               non_basic_coefficients_A,
                                                                               right_hand_side_vector_b, minmax - 1)
        print(f"Basic variables' coefficients: {C_b}")
        print(f"Basic variable numbers: {basic_var_nums}")
        print(f"Basic variable values: {Xb}")
        print(f"Non-basic variable numbers: {non_basic_var_nums}")
        print(f"Decision vector: {decision_vector}")
        f = C_b @ Xb
        f *= 10 ** accuracy
        f = int(f)
        f = float(f) / 10 ** accuracy
        print(f"Objective function value: {f}")
    except Exception as e:
        print("The method is not applicable!")

    break
