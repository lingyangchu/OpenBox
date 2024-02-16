function main()

%%%%%%%% initialize and import data  %%%%%%%%
addpath(genpath('.'));
addpath(genpath('C:\Users\huxiah\Documents\MATLAB\DeepLearnToolbox-master'));
fprintf('\nLoading data......\n');
init = initialize();
[x_sam, y_sam, x_test, y_test] = load_Data(init.dataset);
fprintf('Finish loading %s data. \n', init.dataset);


% k-fold cross validation
k = 10;     % k(mnist56)=8; k(spamemail)=7; k(cifar10)=8
indices = crossvalind('Kfold', size(x_sam, 1), k);
for i = 1 : 1
    clear global model;
    val = (indices == i);
    train = ~val;
    x_train = x_sam(train, :); y_train = y_sam(train);
    x_val = x_sam(val, :); y_val = y_sam(val);
    
    %%%%%%%% train model %%%%%%%%
    train_model(x_train, y_train);
    global model;
    
    %%%%%%%% mkdir and save env %%%%%%%%
    fdir = strcat('result-fold-', num2str(i));
    system(strcat('mkdir', 32, fdir));
    save(strcat(fdir, '/workspace.mat'), 'model');
 
    %%%%%%%% analysis
    if strcmp(init.dataset, 'fashionmnist')
        mnist_bi_analysis
    elseif strcmp(init.dataset, 'multifmnist')
        mnist_multi_analysis
    end
    
end







% ====================================================
function plot_sine()
% PLOT for SINE 
plot_PointsBoundaries(x_train, y_train, tree);
plot_DecisionBoundary(x_train, y_train, tree);
plot_DecisionTree(tree, 'dtlr', fdir);
% plot DNN
figure('Name', 'SINE DNN points');
nn = nnff(nn, x_train, zeros(size(x_train, 1), nn.size(end)));
py_dnn = round(nn.a{end}(:, 2));
plot(x_train(py_dnn==1, 1), x_train(py_dnn==1, 2), 'rx', x_train(~py_dnn, 1), x_train(~py_dnn, 2), 'b+');
figure('Name', 'SINE DNN Decision Boundary');
nn = nnff(nn, x_test, zeros(size(x_test, 1), nn.size(end)));
py_dnn = round(nn.a{end}(:, 2));
plot(x_test(py_dnn==1, 1), x_test(py_dnn==1, 2), 'rx', x_test(~py_dnn, 1), x_test(~py_dnn, 2), 'b+');




%%%%%%%% plot and save result graph %%%%%%%%%   TODO: REWRITE THIS PART 
function analyze()
analyze_DTLR(tree);
if strcmp(data_tag, 'mnist') == 0   % not MNIST
    plot_PointsBoundaries(x_train, y_train, tree);
    plot_DecisionBoundary(x_train, y_train, tree);
else                    % MNIST
    img = ones(1, 784)*0.5;
    global plot_count;
    plot_count = 1;
    plot_TreePathAttr(tree, img); 
    % FOR MNIST: plot original features on each node set
    plot_count = 1;
    plot_Mnist_Treeset(tree, x_train, y_train);
end
% plot decision tree
plot_DecisionTree(dtree, 'dt');
% plot_PointsBoundaries(x_train, y_train, dtree);
% this part is for image
% img = ones(28, 28)*0.5;
% plot_TreePathAttr(dtree, img);


