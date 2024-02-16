function inver_topk_features(x_test, y_test, model)

% reflect_x_by_boundary(x_test, y_test,  model);
move_steps(x_test, y_test, model);

function move_steps(x_test, y_test, model)
k=784;
dl = 'cnn';
if dl=='cnn'
    % for CNN
    net = model.nn;
    net.layers{end-4}.rate = 0;
    py_truth = CNN_test(net, x_test, 1, 'fashionmnist');
elseif dl=='dnn'
    % for DNN
    net = model.nn;
    net = nnff(net, x_test, zeros(size(x_test, 1), net.size(end)));
    py_truth = net.a{end}(:, 2);
end
% filter out correct samples
cor_id = (round(py_truth)==y_test);
x_correct = x_test(cor_id, :);
y_correct = y_test(cor_id);
py_truth = py_truth(cor_id);
n_size = size(x_correct, 1);

% tree=model.tree;
tree = post_prunning(model.tree);
% global gtree; tree=gtree;  

step = 10*ones(n_size, 1);
for i = 1:n_size
    x = x_correct(i, :);
    % DTLR
%     [theta, ~] = traceTreePath(tree, x, 'dtlr');
    % LR
%     theta = model.lr;
    % SVM
    theta = [model.svm.Bias model.svm.Alpha'*model.svm.SupportVectors]';
    % move x on theta's direction
    if [1 x]*theta < 0
        move = [0:0.01:10];
    else
        move = [0:-0.01:-10];
    end
    w = theta(2:end)';
    [~, sortid] = sort(abs(w));  % use top-k features
    w(sortid(1:end-k))=0;
    y = run_deeplearning(net, x, dl);
    id = 0;
    while round(y)==round(py_truth(i)) & id<=1000
        id = id + 1;
        x_moved = x+move(id)*w/(sqrt(sum(w.*w)));
        y = run_deeplearning(net, x_moved, dl);
%         fprintf('%d - %f\n', id, y);
    end
    step(i) = abs(move(id));
    if rem(i, 100)==0
        fprintf('i=%d...\n', i);
        avg_step = sum(step(step~=10))/(sum(step~=10));
        fprintf('average step = %f, unreached = %d \n', avg_step, sum(step==10));
    end
end
avg_step = sum(step(step~=10))/(sum(step~=10));
fprintf('average step = %f, unreached = %d \n', avg_step, sum(step==10));
for a = [0.1:0.2:8]
    fprintf('%d\n', sum(step<=a));
end


% =================================================
function reflect_x_by_boundary(x_test, y_test, model)
n_size = size(x_test, 1);
dl = 'cnn';
if dl=='cnn'
    % for CNN
    net = model.nn;
    net.layers{end-4}.rate = 0;
    py_truth = CNN_test(net, x_test, 1, 'fashionmnist');
elseif dl=='dnn'
    % for DNN
    net = model.nn;
    net = nnff(net, x_test, zeros(n_size, net.size(end)));
    py_truth = net.a{end}(:, 2);
end
% global gtree; tree=gtree; 
tree = post_prunning(model.tree);
py_dtlr = zeros(n_size, 1);
py_dt = zeros(n_size, 1);
py_lr = zeros(n_size, 1);
py_rnd = zeros(n_size, 1);
py_svm = zeros(n_size, 1);
ky=[];
kx=[2,4,6,8,10,16,20,30,40,50,60,70,80,100,150,200,250,300];
kmax=784;
% kx=[1:1:20];
for k = kx
    for i = 1:n_size
       x = x_test(i, :);         

       % filtered feature by each method
       % ===========================================
       % FEATURE FROM DTLR
       % trace path in DTLR tree
       [theta_dtlr, path] = traceTreePath(tree, x, 'dtlr');
       [~, sortid] = sort(abs(path(2:end)));
       topk_dtlr = sortid(end-k+1:end);       
       ref_dtlr = reflect(x, theta_dtlr, topk_dtlr);
       py_dtlr(i) = run_deeplearning(net, ref_dtlr, dl);

       % ===========================================
       % FEATURE FROM LOGISTIC REGRESSION
       theta_lr = model.lr;
       [~, sortid] = sort(abs(theta_lr(2:end)));
       topk_lr = sortid(end-k+1:end);
       ref_lr = reflect(x, theta_lr, topk_lr);
       py_lr(i) = run_deeplearning(net, ref_lr, dl);

       % =========================================
       % FEATURE FROM SVM linear
       theta_svm = [model.svm.Bias model.svm.Alpha'*model.svm.SupportVectors]';
       [~, sortid] = sort(abs(theta_svm(2:end)));
       topk_svm = sortid(end-k+1:end);
       ref_svm = reflect(x, theta_svm, topk_svm);
       py_svm(i) = run_deeplearning(net, ref_svm, dl);
       
       % ===========================================
       % FEATURE FROM RANDOM, without top-k feature of DTLR       
       id = randi([1, kmax-k], k, 1);
       topk_rand = sortid(id);
       ref_rnd = reflect(x, theta_dtlr, topk_rand);
       py_rnd(i) = run_deeplearning(net, ref_rnd, dl);

       % ===========================================
       % FEATURE FROM DECISION TREE
       [~, path] = traceTreePath(model.dtree, x, 'dt');
       ref_dt = x;
       ref_dt(path{1}) = 1;      % left edge, set x's value to be max  16
       ref_dt(path{2}) = 0;      % right edge, set x's value to be min
       py_dt(i) = run_deeplearning(net, ref_dt, dl);
       
    end      
    fprintf('\nk=%d: \n', k);
    fprintf('DTLR Invert amount: %d/%d, accuracy: %d/%d = %f \n',...
                sum(round(py_truth)~=round(py_dtlr)), n_size, sum(round(py_dtlr)==y_test), n_size, sum(round(py_dtlr)==y_test)/n_size);
    fprintf('LR Invert amount: %d/%d, accuracy: %d/%d = %f \n',...
                sum(round(py_truth)~=round(py_lr)), n_size, sum(round(py_lr)==y_test), n_size, sum(round(py_lr)==y_test)/n_size);
    fprintf('SVM Invert amount: %d/%d, accuracy: %d/%d = %f \n',...
                sum(round(py_truth)~=round(py_svm)), n_size, sum(round(py_svm)==y_test), n_size, sum(round(py_svm)==y_test)/n_size);
    fprintf('DT Invert amount: %d/%d, accuracy: %d/%d = %f \n',...
                sum(round(py_truth)~=round(py_dt)), n_size, sum(round(py_dt)==y_test), n_size, sum(round(py_dt)==y_test)/n_size);
    fprintf('RND Invert amount: %d/%d, accuracy: %d/%d = %f \n',...
                sum(round(py_truth)~=round(py_rnd)), n_size, sum(round(py_rnd)==y_test), n_size, sum(round(py_rnd)==y_test)/n_size);
    ky=[ky; sum(round(py_dtlr)==y_test)/n_size,sum(round(py_lr)==y_test)/n_size,...
            sum(round(py_dt)==y_test)/n_size,sum(round(py_rnd)==y_test)/n_size,...
            sum(round(py_svm)==y_test)/n_size];
end
figure;
plot(kx,ky(:,1),'-*', kx,ky(:,2),'-*', kx,ky(:, 3),'-*', kx,ky(:,4),'-*', kx, ky(:,5),'-*');
fprintf(ky);




function x_ref = reflect(x, theta, ref_id)
% move to max or min
x_ref = x;
if [1 x_ref]*theta < 0
    x_ref(ref_id) = (theta(ref_id+1)>0);
    % for spamemail, not normed x
%     x_ref(ref_id) = sign(x_ref(ref_id))'.*((x_ref(ref_id)'.*theta(ref_id+1))>0)*16;
else
    x_ref(ref_id) = (theta(ref_id+1)<0);
    % for spamemail, not normed x
%     x_ref(ref_id) = sign(x_ref(ref_id))'.*((x_ref(ref_id)'.*theta(ref_id+1))<0)*16;
end
% find reflect node
% b = theta(1);
% w = zeros(1, 784);
% w(ref_id)=theta(ref_id+1);
% x_ref = x-2*((sum(w.*x)+b)/sum(w.*w))*w;

function py = run_deeplearning(net, x, type)

if type=='cnn'
    py = CNN_test(net, x, 1, 'fashionmnist');
elseif type=='dnn'
    net = nnff(net, x, [0,0]);
    py = net.a{end}(:, 2);
end



