function gain = Gain_Splitting(x_set, y_set, theta_set)

n = length(x_set);
d_num = size(x_set{1}, 1);
%p1 = zeros(n, 1);
info = zeros(n, 1);

for i = 1 : n
    [accuracy, entropy] = compute_AccuracyEntropy(x_set{i}, y_set{i}, theta_set{i});
    info(i) = size(x_set{i}, 1)*entropy / d_num;
end

gain = info(1) - sum(info(2:n));





