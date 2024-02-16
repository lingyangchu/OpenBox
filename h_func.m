function h = h_func(x, theta)
% we assume x includes the bias
z = x * theta;
h = sigmoid(z);


function g = sigmoid(z)
g = 1./(1+exp(-z));