function [theta, w, J] = Gradient_Descent_Complx(x, y, w, theta, tag)

init = initialize();
x_bias = [ones(size(x, 1), 1), x];

t_start = clock;
i=2;
n_iter = 10000;
J_plot = zeros(n_iter, 1);
J_plot(1:2) = [100, 99];
% update w using SGD
if strcmp(tag, 'w')
    while i < n_iter & abs(J_plot(i-1)-J_plot(i)) > init.grad_j_threshold
        i = i+1;
        [J, djdtheta, djdw] = cost_func(x_bias, y, w, theta);
        w = w - init.lr*djdw;
        J_plot(i) = J;
    end
    
% update theta using SGD
elseif strcmp(tag, 'theta')
    while i < n_iter & abs(J_plot(i-1)-J_plot(i)) > init.grad_j_threshold
        i = i+1;
        [J, djdtheta, djdw] = cost_func(x_bias, y, w, theta);
        theta = theta - init.lr*djdtheta;
        J_plot(i) = J;
    end
end
t_stop = clock;

% for test: compute l1-norm value
theta_l1norm = sum(abs(theta(2:size(theta, 1), :)));
w_l1norm = sum(abs(w(2:size(w, 1)))); 

% figure;
% plot(J_plot(3:i));
fprintf('Update %s: After %d iterations, took %.1f seconds, on %d instances, train err is J = %f, w_l1n = %f, theta_l1n = [%f,%f] \n',...
            tag, i-1, etime(t_stop, t_start), size(x, 1), J, w_l1norm, theta_l1norm(1), theta_l1norm(2));

        
% [theta, J, exitflag] = LR_verify(x_bias, y, S, theta);
% t_stop = clock;
% fprintf('Took %.1f seconds, on %d instances, train err is J = %f \n',...
%             etime(t_stop, t_start), size(x, 1), J);

% function [theta, J, exitflag] = LR_verify(x, y, S, theta)
% 
% options = optimset('GradObj', 'on', 'MaxIter', 500);
% f = @(initTheta)cost_func(x, y, S, initTheta);
% [theta, J, exitflag] = fminunc(f, theta, options);



