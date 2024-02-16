function theta = Logistic_Regression(x, y, init, itheta)

t_start = clock;
x_bias = [ones(size(x,1), 1), x];

% gradient descent update theta
theta = itheta;
J_old = 100;
J = 99;
count = 0;

j_plot = [];
while J_old-J>init.grad_j_threshold & count <= 10000
    J_old = J;
    count = count + 1;
    theta_old = theta;
    
    [J, djdtheta] = cost_func(x_bias, y, ones(size(x, 1), 1), theta);
    theta = theta - init.lr*djdtheta;
    j_plot = [j_plot, J];      
end

t_stop = clock;
fprintf('After %d iterations, took %.1f seconds, on %d instances, train err is J = %f \n',...
            count-1, etime(t_stop, t_start), size(x, 1), J);
% display(theta);
% 
figure;
plot(j_plot);
% This part is to verify our gradient descent result
% [theta_ver, J_ver, exitflag] = LR_verify(x_bias, y, itheta);
% t_stop = clock;
% fprintf('Took %.1f seconds, on %d instances, train err is J = %f \n',...
%             etime(t_stop, t_start), size(x, 1), J_ver);
% fprintf('Logistic Regression result: J = %f exit = %d \n', J_ver, exitflag);
% % display(theta_ver);
% theta = theta_ver;


function [theta, J, exitflag] = LR_verify(x, y, theta)

options = optimset('GradObj', 'on', 'MaxIter', 500);
f = @(initTheta)cost_func(x, y, initTheta);
[theta, J, exitflag] = fminunc(f, theta, options);











