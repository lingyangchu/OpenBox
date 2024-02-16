function [J, djdtheta, djdw]=cost_func(x, y, w, theta)
% % This x contains the bias already
% % x: m * (n+1). eg. [1 x1 x2 x3 ...]
% % theta: (n+1) * K
% % S: assigned matrix, n*k
% % y: m * 1

init = initialize();
n = size(x, 1);
d = size(x, 2);
    
% if strcmp('null', w)            % compute cost function of original theta  
%     p_y1 = h_func([ones(size(x, 1), 1), x], theta);    
%     % J with l1norm
%     J_theta_l1n = (init.lamda_theta)*sum(abs(theta(2:d)));
%     J = (-1/n)*sum(y.*log(p_y1+1e-12) + (1-y).*log(1-p_y1+1e-12)) + J_theta_l1n;  
%     djdtheta = 0; djdw = 0;
% else            % compute cost function of our w-theta split model

% SIGMOID: h(w) & h(theta)
h_w = h_func(x, w);
h_theta = h_func(x, theta);
% p(y=1)
p_y1 = (1-h_w).*h_theta(:, 1) + h_w.*h_theta(:, 2);       % n * 1

dpydtheta1 = repmat((1-h_w).*h_theta(:, 1).*(1-h_theta(:, 1)), 1, d).*x;
dpydtheta2 = repmat(h_w.*h_theta(:, 2).*(1-h_theta(:, 2)), 1, d).*x;      
dpydw = repmat((h_theta(:, 2) - h_theta(:, 1)).*h_w.*(1-h_w), 1, d).*x; 

% djdtheta/w with l1norm
djdtheta1 = (-1/n)*sum(repmat(((y-p_y1+1e-12)./(p_y1.*(1-p_y1) + 1e-12)), 1, d).*dpydtheta1)' ...
                    + (init.lamda_theta)*[0; sign(theta(2:d, 1))];          % djdtheta1's l1-norm
djdtheta2 = (-1/n)*sum(repmat(((y-p_y1+1e-12)./(p_y1.*(1-p_y1) + 1e-12)), 1, d).*dpydtheta2)' ...
                    + (init.lamda_theta)*[0; sign(theta(2:d, 2))];          % djdtheta2's l1-norm
djdtheta = [djdtheta1, djdtheta2];      % d * 2
djdw = (-1/n)*sum(repmat((y-p_y1+1e-12)./(p_y1.*(1-p_y1) + 1e-12), 1, d).*dpydw)'...
        + (init.lamda_w)*[0; sign(w(2:d))];                         % djdw's l1-norm

% J with l1norm
J = (-1/n)*sum(y.*log(p_y1+1e-12) + (1-y).*log(1-p_y1+1e-12)) ...
        + (init.lamda_theta)*sum(sum(abs(theta(2:d, :)))) ...       % J_theta's l1-norm
        + (init.lamda_w)*sum(abs(w(2:d)));                          % J_w's l1-norm





