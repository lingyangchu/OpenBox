function [x_train,y_train, x_test, y_test] = Circle_Sampling()

n = 20000;
d = 3;
x = rand(1.2*n, 2)*d;

x = x - d/2;
x = [x, x(:, 1).^2+x(:, 2).^2];

x(x(:, 3) > 1.0, 3) = 1; 
x(x(:, 3) < 1, 3) = 0;

selected_id = (x(:, 3) == 0 | x(:, 3) == 1);
x = x(selected_id, :);

y_train = x(1:4000, 3);
x_train = x(1:4000, 1:2);
x_test = x(4001:n, 1:2);
y_test = x(4001:n, 3);
% x_test = rand(10*n, 2)*d - d/2;
% y_test = [];

figure('name','train');
plot(x_train(y_train==1, 1), x_train(y_train==1, 2), 'r.', x_train(y_train==0, 1), x_train(y_train==0, 2), 'b.');
figure('name','test');
plot(x_test(y_test==1, 1), x_test(y_test==1, 2), 'r.', x_test(y_test==0, 1), x_test(y_test==0, 2), 'b.');



