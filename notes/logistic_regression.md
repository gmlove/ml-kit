Known:
y = f(x1, x2)
y in [0, 1]
sigmoid function: y = 1 / (1 + exp(-x))

Goal:
find the relationship between sigmoid function and f(x1, x2, ...) when x=0 in sigmoid function

Inference:
part of (x1, x2): g(x1, x2) = 0; other part of (x1, x2): g(x1, x2) = 1
g(x1, x2) = 1 / (1 + exp(h(x1, x2)))
assumption: h(x1, x2) = w0 + w1 * x1 + w2 * x2
g(x1, x2) = 1 / (1 + exp(w0 + w1 * x1 + w2 * x2))
Need to find vector W. We can use known data to achieve this.
set W = ones((n, 1))
E = g(X) - 1 / (1 + exp(W*X))
goal is to find W when E is limited to 0
set W = W + alpha * X.T * E, and loop a number of times, the W will be found


reference:
http://www.cnblogs.com/jerrylead/archive/2011/03/05/1971867.html
http://www.cnblogs.com/LeftNotEasy/archive/2010/12/05/mathmatic_in_machine_learning_1_regression_and_gradient_descent.html