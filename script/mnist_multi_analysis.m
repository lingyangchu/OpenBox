% analysis of mnist_multi classification
% regular 2 hidden layer

% TAG
PLOT_CONFIG = true;
PLOT_TRANSFORMATION = true;
ANALYZE_EXACT_SAMPLE = false;
ANALYZE_NEURON_FUNCTION = false;

x_total = [x_sam; x_test];
y_total = [y_sam; y_test];
n_size = size(x_total, 1);
nn = nnff(model.nn, x_total, zeros(n_size, model.nn.size(end)));
[~, py_nn] = max(nn.a{end}, [], 2);
mnist.uy = sort(unique(y_total));
fprintf('Accuracy = %.4f%% \n', 100*sum(mnist.uy(py_nn)==y_total)/n_size);

% generate neuralhit 
neuralhit = [];
for i=[2:nn.n-1]
    neuralhit = [neuralhit, (nn.a{i}(:, 2:end))>0];
end
mnist.neuralhit = neuralhit;
% find unique neuralhits
[mnist.neuralhit_unique, ~, mnist.loc] = unique(neuralhit, 'rows');
fprintf('In total %d unique neural hits. \n', size(mnist.neuralhit_unique, 1));



% ================ COMPUTE CONFIGURATION MATRIX =============
% compute matrix
config_matrix_w = zeros(size(mnist.neuralhit_unique, 1), 784);
config_matrix_b = zeros(size(mnist.neuralhit_unique, 1), 1);
w_1 = nn.W{1}(:, 2:end); b_1 = nn.W{1}(:, 1); 
w_2 = nn.W{2}(:, 2:end); b_2 = nn.W{2}(:, 1); 
w_output = nn.W{end}(:, 2:end); b_output = nn.W{end}(:, 1);
W={}; B={};
for i = [1:size(mnist.neuralhit_unique, 1)]
    % compute w1 and b2
    relu1 = mnist.neuralhit_unique(i, 1:size(b_1, 1));
    w_1r=w_1; b_1r = b_1;
    w_1r(relu1==0, :) = 0; b_1r(relu1==0, :) = 0;
    % compute w2 and b2
    relu2 = mnist.neuralhit_unique(i, (size(b_1, 1)+1):end);
    w_2r=w_2; b_2r=b_2;
    w_2r(relu2==0, :) = 0; b_2r(relu2==0, :) = 0;
    % compute fw and fb for whole net
    fw = w_output*w_2r*w_1r; fb = w_output*w_2r*b_1r + w_output*b_2r + b_output;
    % plot fw
    W{i}=fw;B{i}=fb;
    
    config_matrix_w(i, :) = fw(1, :)-fw(2, :);
    config_matrix_b(i, :, :) = fb(1)-fb(2);
end

% compute similarity matrix of config_matrix
mnist.config_matrix = [config_matrix_b config_matrix_w];
mnist.similarity_mat = consine_similarity(mnist.config_matrix);

% =================================
if PLOT_CONFIG
    % for given x, plot [features and data] of this configuration
    % given x ==> find configuration & group ==> plot samples & features
    % k=3001;
    % % select data in group
    % id = loc(k);    % which group this belongs to
    for id = [1:size(mnist.neuralhit_unique, 1)]
        x_group = x_total(mnist.loc==id,:);
        py_group = py_nn(mnist.loc==id, :);
        distribute = [sum(py_group==1), sum(py_group==2), sum(py_group==3), sum(py_group==4)];
        fprintf('This configuration group: %d = (%d : %d : %d : %d)\n', size(x_group, 1), distribute);
        % get w and b 
        w_group = squeeze(config_matrix_w(id, :));
        b_group = squeeze(config_matrix_b(id, :));
        % plot features and images
        figure('name', sprintf('%d(%d,%d,%d,%d)', size(x_group, 1), distribute));
        for i=1:length(mnist.uy)
            subplot(4,4,i*2-1);
            shape0 = sum(x_group(py_group==i, :), 1)/sum(py_group==i);
            imshow(reshape(shape0, 28, 28));
            subplot(4, 4, i*2); 
            imshow(reshape(abs(W{id}(i, :)), 28, 28));
        end
        colormap(jet);
    end
end


% =======================================================
% solve neuron functions, and compute redundant constraint
if ANALYZE_NEURON_FUNCTION
    w_1 = nn.W{1}(:, 2:end); b_1 = nn.W{1}(:, 1); 
    w_2 = nn.W{2}(:, 2:end); b_2 = nn.W{2}(:, 1); 
    w_output = nn.W{end}(:, 2:end); b_output = nn.W{end}(:, 1);
    bound = [];
    for i = [1:size(mnist.neuralhit_unique, 1)]
        % first layer, w is original network's w
        % second layer, w computed from first layer
        if strcmp(nn.activation_function, 'relu')   
            relu1 = neuralhit_unique(i, 1:size(b_1, 1))';
            w2 = w_2*(w_1.*repmat(relu1, 1, size(w_1, 2))); 
            b2 = w_2*(b_1.*relu1)+b_2; 
            w = [w_1; w2];
            b = [b_1; b2];
        elseif strcmp(nn.activation_function, 'relu-nonneg')
            relu1 = mnist.neuralhit_unique(i, 1 : size(w_1, 1))';
            w2 = w_2*(w_1.*repmat(relu1, 1, size(w_1, 2)));
            b2 = w_2*((b_1-nn.relu_thres).*relu1)+b_2;
            w = [w_1; w2];
            b = [b_1; b2]-nn.relu_thres;
        end


        % if act == 1, -w, b; else w, -b
        relu = mnist.neuralhit_unique(i, :);
        w(relu==1, :) = -1*w(relu==1, :);
        b(~relu, :) = -1*b(~relu, :);       % we can plot w to see how these features look like

        thisbound = linear_redundant_constraints(w, b, zeros(size(w, 2), 1), ones(size(w, 2), 1));
        thisbound = thisbound.*(relu-0.5)*2; 
        bound = [bound; thisbound];
    end
end
    
    



