function [x_sam, y_sam] = Sine_Sampling()

pot = 0:0.2:4*pi;
dist = 0.7;
sigma = [0.03, 0; 0, 0.03];
x_beforer = [];
y_beforer = [];
n_rndpoint = 20;

for i = pot
    rng default
    mu_pos = [i, sin(i) + dist];
    r_pos = mvnrnd(mu_pos, sigma, n_rndpoint);
    x_beforer = [x_beforer; r_pos];
    y_beforer = [y_beforer; ones(n_rndpoint, 1)];
    
    mu_neg = [i, sin(i) - dist];
    r_neg = mvnrnd(mu_neg, sigma, n_rndpoint);
    x_beforer = [x_beforer; r_neg];
    y_beforer = [y_beforer; zeros(n_rndpoint, 1)];
end

% pick 1:2400 part from x_beforer
x_beforer = x_beforer(1:2400, :);
y_beforer = y_beforer(1:2400);

% rand perm
x_sam = zeros(size(x_beforer));
y_sam = zeros(size(y_beforer));
id_rand = randperm(size(x_sam, 1));
for i = 1:size(x_beforer, 1)
    x_sam(i, :) = x_beforer(id_rand(i), :);
    y_sam(i) = y_beforer(id_rand(i));
end

pos = (y_sam == 1);
neg = ~pos;
figure;
plot(x_sam(pos, 1), x_sam(pos, 2), 'r+', x_sam(neg, 1), x_sam(neg, 2), 'b+');








