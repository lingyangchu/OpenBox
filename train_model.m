function train_model(x_train, y_train)

global model;

% SET PARAMETERS FOR TRAINING
init = initialize();

% ======================= DEEP LEARNING ====================
fprintf('\n****************************************************\n');
if strcmp(init.dl_model, 'dnn')
    fprintf('Begin training DNN: \n');
    t_start = clock;
    % train a signed network
    [py_nn, nn] = DNN(x_train, y_train, 0, {});
    model.negnn = nn;
    % train a non-negative network
    initw = nn.W;
    [py_nn, nn] = DNN(x_train, y_train, 1, initw);
    model.nonnegnn = nn;
    t_stop = clock;
elseif strcmp(init.dl_model, 'cnn')
    fprintf('Begin training CNN: \n');
    t_start = clock;
    [py_nn, nn] = CNN(x_train, y_train, init.dataset);
    t_stop = clock;
    py_nn = py_nn(2, :)';
end
model.nn = nn;
uy = sort(unique(y_train));
fprintf('Finish training %s. Accuracy on training data is %.4f, Running cost: %f min \r\n',...
                init.dl_model, sum(uy(py_nn)==y_train)/size(y_train, 1),etime(t_stop, t_start)/60);


% =========================================
%           NON-NEGATIVE LR
fprintf('\n****************************************************\n');
fprintf('Begin training LR-nonneg: \n');
theta = Logistic_Regression_nonneg(x_train, y_train);
model.lr = theta;
fprintf('Finish training LR-nonneg model.\r\n');













