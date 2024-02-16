% train several models with same structure and same input data
% plot and compare the boundary, and transformed space
% dataset: circle

% with different random initialize value of w and b
for i = 1:5
    [~, nn(i)] = DNN(x_train, y_train);
    model.nn = nn(i);
    plot_relu
end

% use same init value of w and b
n = size(x_train, 2);m = size(x_train, 1);
nn = nnsetup([n 8*n n 2]);
nn.output = 'softmax';
opts.numepochs = 100;
opts.batchsize = 100;
train_y = zeros(m, 2);
for i = 1:m
   train_y(i, y_train(i, 1)+1) = 1; 
end
for i = 1:5
    [model.nn, ~] = nntrain(nn, x_train, train_y, opts);
    plot_relu
end