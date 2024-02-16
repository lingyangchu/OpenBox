function result = test_model(x_sam, y_sam, fid, model, tag)

init = initialize();
n_size = size(x_sam, 1);

fprintf(fid, '\nOn %s data:\n', tag);
% % ***************   DECISION TREE
% y_dtree = zeros(n_size, 1);
% for i = 1:n_size
%    x = x_sam(i, :);
%    [theta, path] = traceTreePath(model.dtree, x, 'dt');
%    y_dtree(i) = theta;
% end
% [accuracy, recall, precision, fmeasure] = evaluation(y_dtree, y_sam);
% result.dtree = [accuracy, recall, precision, fmeasure];
% fprintf(fid, '\tDecision Tree: [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n',...
%             sum(y_dtree==y_sam), n_size, accuracy, recall, precision, fmeasure);

        
% ***************   SVM
% y_svm = svmclassify(model.svm, x_sam);
% y_svm = 0.5 + (y_svm/2);
% [accuracy, recall, precision, fmeasure] = evaluation(y_svm, y_sam);
% result.svm = [accuracy, recall, precision, fmeasure];
% fprintf(fid, '\tSVM: [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n', ...
%             sum(y_svm==y_sam), n_size, accuracy, recall, precision, fmeasure);

        
% % ***************   ADABOOST
% y_ada = adaboost('apply', x_sam, model.ada);
% y_ada = 0.5 + (y_ada/2);
% [accuracy, recall, precision, fmeasure] = evaluation(y_ada, y_sam);
% result.adaboost = [accuracy, recall, precision, fmeasure];
% fprintf(fid, '\tAdaBoost: [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n', ...
%             sum(y_ada==y_sam), n_size, accuracy, recall, precision, fmeasure);
%         
%         
% % ***************   LOGISTIC REGRESSION
lr_theta = model.lr;
py_lr = h_func([ones(size(x_sam, 1), 1), x_sam], lr_theta);
[accuracy, recall, precision, fmeasure] = evaluation(py_lr, y_sam);
result.lr = [accuracy, recall, precision, fmeasure];
% fprintf(fid, '\tLogistic Regression:  [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n', ...
%             sum(round(py_lr)==y_sam), n_size, accuracy, recall, precision, fmeasure);
% 
% 
% % % ***************   DPMM
% % [~, py_dpmm] = CL_predict(x_sam,  model.cl);
% % [accuracy, recall, precision, fmeasure] = evaluation(py_dpmm, y_sam);
% % result.dpmm = [accuracy, recall, precision, fmeasure];
% % fprintf(fid, '\tDPMCM: [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n', ...
% %             sum(round(py_dpmm)==y_sam), n_size, accuracy, recall, precision, fmeasure);
% %     
%                 
% % ***************   DEEP LEARNING
% if strcmp(init.dl_model, 'dnn')
%     % compute DNN
%     nn = nnff(model.nn, x_sam, zeros(n_size, model.nn.size(end)));
%     py_nn = nn.a{end}(:, 2);
% elseif strcmp(init.dl_model, 'cnn')
%     [py_nn, ~] = CNN(x_sam, y_sam, init.dataset);
%     py_nn = py_nn(2, :)';
% end
% [accuracy, recall, precision, fmeasure] = evaluation(py_nn, y_sam);
% result.deep_model = [accuracy, recall, precision, fmeasure];
% fprintf(fid, '\tDNN: [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n',...
%             sum(round(py_nn)==y_sam), n_size, accuracy, recall, precision, fmeasure);

        
        
% ***************   DTLR MODEL
py_dt = zeros(n_size, 1);
for i = 1:n_size
   x = x_sam(i, :);
   [theta, path] = traceTreePath(model.tree, x, 'dtlr');
   py_dt(i) = h_func([1, x], theta);
end
[accuracy, recall, precision, fmeasure] = evaluation(py_dt, y_sam);
result.dtlr = [accuracy, recall, precision, fmeasure];
fprintf(fid, '\tDTLR: [%d/%d, acc=%.4f, recall=%.4f, precision=%.4f, f-measure=%.4f]\n',...
            sum(round(py_dt)==y_sam), n_size, accuracy, recall, precision, fmeasure);
        
match_num = sum(round(py_nn)==round(py_dt));
l2_dist = (sum((py_dt - py_nn).^2))^.5;
result.match_deepmodel = [match_num/n_size, l2_dist];
fprintf(fid, '\tDTLR match DNN: [%d/%d, L2-dist: %.8f]\n', ...
            match_num, n_size, l2_dist);

plot(x_test(py_dt==1, 1), x_test(py_dt==1, 2), 'r.', x_test(py_dt==0, 1), x_test(py_dt==0, 2), 'b.')
     



function [accuracy, recall, precision, fmeasure] = evaluation(py, gtruth)

tp = sum(round(py) & gtruth);
fn = sum(round(py)==0 & gtruth);
fp = sum(round(py) & gtruth==0);
tn = sum(round(py)==0 & gtruth==0);

accuracy = (tp+tn)/(tp+tn+fp+fn);
recall = tp/(tp+fn);
precision = tp/(tp+fp);
fmeasure = (2*tp)/(2*tp+fn+fp);

% =========================================      
%         CODE FOR GEN DATA PLOT
% =========================================
function gen_data_plot()
x_total = [x_sam; x_test];
y_total = [y_sam; y_test];
figure('name', 'basic');
plot(x_total(y_total==0, 1), x_total(y_total==0, 2), 'r.', ...
        x_total(y_total==1, 1), x_total(y_total==1, 2), 'b.');

n_size = size(x_total, 1);
nn = nnff(model.nn, x_total, zeros(n_size, model.nn.size(end)));
py_nn = nn.a{end}(:, 2);
figure; scatter(x_total(:, 1), x_total(:, 2), [], round(py_nn, 1));
py_nn = round(py_nn);
figure('name', 'dnn');
plot(x_total(py_nn==0, 1), x_total(py_nn==0, 2), 'r.', ...
        x_total(py_nn==1, 1), x_total(py_nn==1, 2), 'b.');

py_dt = zeros(n_size, 1);
for i = 1:n_size
   x = x_total(i, :);
   [theta, path] = traceTreePath(model.tree, x, 'dtlr');
   py_dt(i) = h_func([1, x], theta);
end
figure; scatter(x_total(:, 1), x_total(:, 2), [], round(py_dt, 1));
py_dt = round(py_dt);
figure('name', 'DTLR');
plot(x_total(py_dt==0, 1), x_total(py_dt==0, 2), 'r.', ...
        x_total(py_dt==1, 1), x_total(py_dt==1, 2), 'b.');
figure('name', 'err');
err = ~(py_dt==py_nn);
plot(x_total(err, 1), x_total(err, 2), 'r.');        

% ============= for test; filter matrix of w/theta -- for fashion-mnist
function Fashion_mnist_test()

for i = 1:28
    for j = 1:28
        i_begin = max(i-1, 1); i_end = min(i+1, 28);
        j_begin = max(j-1, 1); j_end = min(j+1, 28);
        cur = w(i_begin:i_end, j_begin:j_end);
        s(i,j) = sum(sum(cur));
        n(i,j) = sum(sum(cur~=0));
        if s(i,j) < 1.2 | n(i,j) < 4
            w(i,j)=0;
        end
    end
end

% ---------------------------

