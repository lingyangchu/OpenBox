function L = log_likelihood(x, y, theta)

p1 = h_func([ones(size(x, 1), 1), x], theta);
L = sum(y.*log(p1+1e-12) + (1-y).*log(1-p1+1e-12));

% MTHOD 3: DECISION TREE MAX-LIKELIHOOD MODEL
% % compute the max-likelihood based on current division
% k = size(S, 2);
% L = sum(sum(S.*(repmat(y, 1, k).*log(p1+1e-10)+repmat((1-y), 1, k).*log(1-p1+1e-10))));


