function [accuracy, info, l2dist] = compute_AccuracyEntropy(x, y, theta)
%% This function is to compute the accuracy, entropy and L2-distance of a LR MODEL
%% x: training data set
%% y: class label
%% theta: theta of the LR model

x_bias = [ones(size(x,1), 1), x];
h = h_func(x_bias, theta);

% compute accuracy
n_accur = sum(round(y) == round(h));
accuracy = n_accur / size(x, 1);

% compute entropy
if accuracy == 1 || accuracy == 0
    info = 0;
else
    info = -( accuracy*log(accuracy) + (1-accuracy)*log(1-accuracy) );
end

% compute l2 distance
l2dist = (sum((h - y).^2))^.5;
