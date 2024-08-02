import numpy as np

"""

1. Find the equation of the line
(0, 3) and (-2, 0) --> m = 3/2
x_2 = 3/2x_1 + 3

2. Find the equation of the line in standard form. --> w1x1 + w2x2 + w0 = 0 *NOTEEEE
0 =  x_2 - 3/2x_1 - 3 OR 0 = 3/2x_1 - x_2 + 3

(Note sure... 3. Use graph to determine the correct line equation
Plug in to any point on graph, which shows sign (i.e. 0,0 on negative side)
==> The only equation that is true is 0 =  2x_2 - 3x_1 - 6)

*NOTE 
The perceptron finds the weighted sum (z) of the inputs plus the bias and inputs that to the activation 
function.
z = w1x1 + w2x2 + ..... + w0 
Typical activation functions output 1 for positive values of z and 0 for negative values for z.
Thus the threshold for z is 0 --> so thats why "w1x1 + w2x2 + w0 = 0" represents the decision boundary 
for a perceptron.
"""

weights = np.array([-3/2, 1, 3])
