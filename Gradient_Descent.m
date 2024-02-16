function theta = Gradient_Descent(x, y, theta)

init = initialize();
x_bias = [ones(size(x, 1), 1), x];

t_start = clock;
i = 2;
n_iter = 2000;
n = size(x, 1);
d = size(x_bias, 2);
J_plot = zeros(n_iter, 1);
J_plot(1:2) = [100, 99];
while i < n_iter & abs(J_plot(i-1) - J_plot(i)) > init.grad_j_threshold
    i = i+1;
    % compute J and djdtheta
    p1 = h_func(x_bias, theta);
    % compute J
    J_l1norm = (init.lamda_theta) * sum(abs(theta(2:d)));
    J = (-1/n)*sum(y.*log(p1+1e-12) + (1-y).*log(1-p1+1e-12)); % + J_l1norm;
    djdtheta_l1norm = (init.lamda_theta)*[0; sign(theta(2:d))];
    djdtheta = (-1/n)* sum(repmat((y-p1), 1, d).*x_bias)'; % + djdtheta_l1norm;
    % update theta
    theta = theta - 0.01*djdtheta;
    J_plot(i) = J;
end
t_stop = clock;

figure;
plot(J_plot(3:i));
fprintf('After %d iterations, took %.1f seconds, on %d instances, train err is J = %f \n', ...
            i-1, etime(t_stop, t_start), size(x, 1), J);



        


