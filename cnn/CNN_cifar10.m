function [py_cnn, net] = CNN_cifar10(x_sam, y_sam)

% CNN for cifar 10
opts.expDir = fullfile(vl_rootnn, 'data', 'my-cifar');
opts.train.gpus = [];
mkdir(opts.expDir);

% BUILD IMDB
x = permute(x_sam, [2 1]);
x = reshape(x, [32 32 1 size(x_sam, 1)]);
imdb.images.data = single(x);
dataMean = mean(imdb.images.data, 4);       % data - mean
% imdb.images.data = single(bsxfun(@minus, imdb.images.data, dataMean));
imdb.images.data_mean = dataMean;
imdb.images.labels = (y_sam+1)';    % matconvnet cannot recognize label 0
imdb.images.set = [ones(length(y_sam)-1000, 1); 3*ones(1000, 1)];
imdb.meta.sets = {'train', 'val', 'test'};

% TRAINING
net = cnn_cifar10_init();
[net, info] = cnn_train( net, imdb, getBatches(opts), ...
                    'expDir', opts.expDir, ...
                    net.meta.trainOpts, ...
                    opts.train, ...
                    'val', find(imdb.images.set == 3) ...
                    );
     
             
% MUST USE TEST DATA MINUS TRAIN_DATA_MEAN AS INPUT
net.layers{end}.type = 'softmax';
net.layers{end-2}.rate = 0;
net.dataMean = dataMean;
res = vl_simplenn(net, imdb.images.data);
py_cnn = squeeze(res(end).x);
fprintf('CNN accuracy on training data is: %f \r\n', sum(y_sam == round(py_cnn(2, :)'))/8000);

    
function fn = getBatches(opts)
fn = @(x, y)getSimpleNNBatch(x, y);

function [images, labels] = getSimpleNNBatch(imdb, batch)
images = imdb.images.data(:,:,:, batch);
labels = imdb.images.labels(1, batch);



function net = cnn_cifar10_init()
% design LENET5 structure for CIFAR-10
lr = [.1 .2];
net.layers = {};
                
% conv 1
net.layers{end+1} = struct('type', 'conv', ...
                            'weights', {{0.01*randn(5, 5, 1, 64, 'single'), zeros(1, 64, 'single')}}, ...
                            'learningRate', lr, ...
                            'stride', 1, ...
                            'pad', 2 ...
                    );
net.layers{end+1} = struct('type', 'pool', ...
                            'method', 'max', ...
                            'pool', [3 3], ...
                            'stride', 2, ...
                            'pad', [0 1 0 1] ...
                    );
net.layers{end+1} = struct('type', 'relu');
% conv 2
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(5,5,64,64, 'single'), zeros(1, 64,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; % Emulate caffe
% conv 3
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(5,5,64,192, 'single'), zeros(1,192,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 2) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', [0 1 0 1]) ; % Emulate caffe
% fully connected
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(4,4,192,64, 'single'), zeros(1,64,'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5);
% output layer
net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,64,2, 'single'), zeros(1,2,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;                     
% Loss layer
net.layers{end+1} = struct('type', 'softmaxloss') ;

% % meta parameters
net.meta.inputSize = [32 32 1] ;
net.meta.trainOpts.learningRate = [0.05*ones(1,100) 0.005*ones(1,100) 0.0005*ones(1,100)] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;
