function tree = Decision_Tree(x_sam, y_sam, cur_theta)

% create a tree node & set value
tree = struct('w','null','J', 'null', 'l2dist', 'null',...
                'theta','null','accuracy','null','info','null',...
                'num_data','null','num_p1','null',...
                'cost', 'null',...
                'left','null', 'right','null');
tree.theta = cur_theta;
tree.num_data = length(y_sam);
tree.num_p1 = sum(round(y_sam));

% compute cost value
% [tree.J, ~, ~] = cost_func(x_sam, y_sam, 'null', cur_theta);
init = initialize();
p_y1 = h_func([ones(size(x_sam, 1), 1), x_sam], cur_theta);    
% J with l1norm
J_theta_l1n = (init.lamda_theta)*sum(abs(cur_theta(2:end)));
tree.J = (-1/size(x_sam, 1))*sum(y_sam.*log(p_y1+1e-12) + (1-y_sam).*log(1-p_y1+1e-12)) + J_theta_l1n;  


[tree.accuracy, tree.info, tree.l2dist] = compute_AccuracyEntropy(x_sam, y_sam, cur_theta);


% stop splitting threshold, mark this as a leaf node
%   1. accuracy > threshold
if tree.accuracy == 1
    fprintf('This is a leaf node. On %d instances, accuracy = 1, train err is J = %f \n\n', tree.num_data, tree.J);
    return;
end
if tree.num_data == 1
    display('This is a leaf node. Instance number == 1. \n\n');
    return;
end

% PART 1: INIT W AND THETA
n = size(x_sam, 1);
d = size(x_sam, 2) + 1;
k = 2;
w = normrnd(0, 0.1, [d, 1]);
lr_theta = normrnd(0, 0.1, [d, k]);
count = 1;

% PART 2: LOOP AND FIND BEST W AND THETA
J_old = 100;
J = 99;
while J_old - J > init.w_theta_alteroptimize
    J_old = J;
    count = count + 1;
    % optimize theta: SGD
    [lr_theta, w, J] = Gradient_Descent_Complx(x_sam, y_sam, w, lr_theta, 'theta');
    % optimize w: SGD
    [lr_theta, w, J] = Gradient_Descent_Complx(x_sam, y_sam, w, lr_theta, 'w');   
end


% PART 3: SPLIT INTO 2 NODE AND CALL DECISION_TREE
% loglike_split = log_likelihood(x_sam, y_sam, lr_theta, w);
J_split = J;
S = Split_Dataset(x_sam, w); 
if J_split < tree.J & sum(sum(S) == 0)==0
    fprintf('After %d iterations, J = %e > J_split = %e. J_split - J = %f. Split into two nodes.\r\n', ...
                count, tree.J, J_split, (J_split - tree.J));
    tree.w = w;
    tree.J = J_split; 
    
    tree.left = Decision_Tree(x_sam(S(:, 1), :), y_sam(S(:, 1)), lr_theta(:, 1));
    tree.right = Decision_Tree(x_sam(S(:, 2), :), y_sam(S(:, 2)), lr_theta(:, 2));
else
    fprintf('After %d iterations, J = %e < J_split = %e. This is a leaf node.\r\n', count, tree.J, J_split);
end






