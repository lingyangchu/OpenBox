function theta = Logistic_Regression_nonneg(x, y)

theta =  [0; exprnd(0.001, size(x, 2), 1)];
x_bias = [ones(size(x, 1), 1), x];

lambda = 0.01;
lr = 0.2;
j_thres = 1e-10;
i = 2;
n_iter = 5000;
n = size(x, 1);
d = size(x_bias, 2);
J_plot = zeros(n_iter, 1);
J_plot(1:2) = [100, 99];
while i < n_iter & abs(J_plot(i-1) - J_plot(i)) > j_thres
    i = i+1;
    % compute J and djdtheta
    p1 = h_func(x_bias, theta);
    % compute J
    J_l1norm = lambda * sum(abs(theta(2:d)));
    J = (-1/n)*sum(y.*log(p1+1e-12) + (1-y).*log(1-p1+1e-12)); 
    djdtheta_l1norm = lambda*[0; sign(theta(2:d))];
    djdtheta = (-1/n)* sum(repmat((y-p1), 1, d).*x_bias)';
    % update theta
    theta = theta - lr*djdtheta;
    theta(2:end) = max(0, theta(2:end));
    J_plot(i) = J;
end

figure;
plot(J_plot(3:i));
fprintf('After %d iterations, on %d instances, acc = %f, train err is J = %f \n', ...
                i-1, size(x, 1), sum(round(p1)==y)/size(y, 1), J);
