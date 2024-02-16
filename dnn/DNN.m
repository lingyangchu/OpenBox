function [py, nn] = DNN(x_sam, y_sam, isnonneg, preparam)
%% y_sam = 0/1

init = initialize();
train_x = x_sam;
m = size(train_x, 1);
n = size(train_x, 2);

% spread y_sam to be a multi class one-hot matrix
n_label = length(unique(y_sam));
uy = sort(unique(y_sam));
train_y = zeros(m, n_label);
for i = 1:n_label
    id = (y_sam==uy(i));
    train_y(id, i)=1;
end


if strcmp(init.dataset, 'gaussian')
    % gaussian forward pass:
    nn = nnsetup([n 8*n+1 2*n 2]);
    opts.numepochs = 400;
    opts.batchsize = 20;    
elseif strcmp(init.dataset, 'circle')
    % circle sampling forward pass:[n 8*n 2]
    nn = nnsetup([n 8*n n 2]);
    nn.output = 'softmax';
    opts.numepochs = 100;
    opts.batchsize = 100;
elseif strcmp(init.dataset, 'animalshape')
    nn = nnsetup([n 16*n 8*n+1 n 2]);        % [n 8*n 2]
    opts.numepochs = 200;
    opts.batchsize = 313;   %156 for shape1, 130 for shape2, 137 for shape3
elseif strcmp(init.dataset, 'breastcc')
    % breastcc forward pass: 2 hidden layer & 1 prediction layer
    nn = nnsetup([n 2*n 4*n 2]);
    opts.numepochs = 50;
    opts.batchsize = 50;
elseif strcmp(init.dataset, 'sine')
    % sine sampling forward pass:
    nn = nnsetup([n 4*n+1 2]);
    opts.numepochs = 600;
    opts.batchsize = 40;
elseif strcmp(init.dataset, 'spamemail')
    % % spamemail forward pass: 2 hidden layer & 1 prediction layer [n 100 28 2]
    nn = nnsetup([n 4*n 2*n n+1 n/2 2]);
    opts.numepochs = 400;
    opts.batchsize = 25;
elseif strncmp(init.dataset, 'mnist', 5)
    % MNIST forward pass: 1 hidden layer, ReLU & 1 prediction layer, softmax
    nn = nnsetup([n 10 2 2]);        % fashion-mnist, 100. sigm: [n 100 2 2]
    nn.output = 'softmax';
    opts.numepochs = 200;
    opts.batchsize = 10;        % 0&5: 16; 1&7: 50
elseif strcmp(init.dataset, 'fashionmnist')
%     nn = nnsetup([n 15 4 2]);        % fashion-mnist, 100. sigm: [n 100 2 2]
    nn = nnsetup([n 8 2 2]);
    nn.output = 'softmax';
    
    opts.numepochs = 200;
    opts.batchsize = 10;
    if isnonneg~=0
        nn.activation_function = 'relu-nonneg';
    else
        nn.activation_function = 'relu';
        opts.numepochs = 100;
    end
elseif strcmp(init.dataset, 'multifmnist')
    nn = nnsetup([n 20 8 n_label]);
    nn.output = 'softmax';
    nn.activation_function = 'relu-nonneg';
%     nn.learningRate = 0.2;
    opts.numepochs = 300;
    opts.batchsize = 100;
    if strcmp(nn.activation_function, 'relu-nonneg')
        opts.batchsize = 100;
    end
    
elseif strcmp(init.dataset, 'moviereview')
    nn = nnsetup([n 500 100 20 2]);
    nn.output = 'softmax';
    opts.numepochs = 20;
    opts.batchsize = 1000;
    
    if isnonneg ~= 0
        nn.activation_function = 'relu-nonneg';
    else
        nn.activation_function = 'relu';
    end
    
elseif strcmp(init.dataset, 'ck15')
    nn = nnsetup([n n/4 n/16 n/64 n/256 2]);
    opts.numepochs = 50;
    opts.batchsize = 1;    
end

% set pre-trained weight
if length(preparam) ~= 0
    for i = 1:nn.n-1
        nn.W{i} = preparam{i};
    end
end

% training: L is cost result for each minibatch epoch
[nn, L] = nntrain(nn, train_x, train_y, opts);
% testing
% [er, bad] = nntest(nn, train_x, train_y);

% save result to a file
nn = nnff(nn, train_x, zeros(size(train_x, 1), nn.size(end)));
[~, py] = max(nn.a{end}, [], 2);







