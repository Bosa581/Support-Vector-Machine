#Orobosa Igbinovia
from numpy import array, zeros

# Data: kINSP and kSEP
# Two datasets with three columns representing (x_1, x_2, y). 
# The first two values are the 2D features, and the third value
# is the label, where +1 and -1 indicate class membership.
# In these datasets:
# x = dataset[:, :-1]
# y = dataset[:, -1] 
kINSP = array([(1, 8, +1),
               (7, 2, -1),
               (6, -1, -1),
               (-5, 0, +1),
               (-5, 1, -1),
               (-5, 2, +1),
               (6, 3, +1),
               (6, 1, -1),
               (5, 2, -1)])

kSEP = array([(-2, 2, +1),    # 0 - A
              (0, 4, +1),     # 1 - B
              (2, 1, +1),     # 2 - C
              (-2, -3, -1),   # 3 - D
              (0, -1, -1),    # 4 - E
              (2, -3, -1),    # 5 - F
              ])

def weight_vector(x, y, alpha):
    """
    Given a vector of alphas, compute the weight vector.
    """

    w = zeros(len(x[0]))  # Initialize the weight vector
    
    '''
    Implement the logic to compute the weight vector.
    Hint: Loop over the data points and accumulate the contributions to the weight vector.
    '''  
 
    for i in range(len(x)):
       for j in range(len(x[i])):
           w[j] += (x[i][j]* alpha[i] * y[i])
    return w


def find_support(x, y, w, b, tolerance=0.001):
    """
    Return the indices for all of the support vectors.
    """
    
    support = set()
    
    '''
    Implement the logic to identify support vectors.
    Hints: what should each support vector satisfy? Use the tolerance value to handle precision issues.
    '''

    for i in range(len(x)):
        total  = 0
        for j in range(len(x[i])):
            total += (x[i][j] * w[j])
        if y[i] >= (total + b) - tolerance and y[i] <= (total + b) + tolerance:
            support.add(i)
    
    return support


def find_slack(x, y, w, b):
    """
    Return the indices for all of the slack vectors.
    """
    
    slack = set()
    
    '''
    Implement the logic to identify slack vectors.
    Hint: Slack vectors violate the margin constraint.
    '''
    for i in range(len(x)):
        total = 0
        for j in range(len(x[i])):
            total +=(x[i][j] * w[j])
        score = total + b
        if y[i] * score < 1:
            slack.add(i)
    
    return slack
