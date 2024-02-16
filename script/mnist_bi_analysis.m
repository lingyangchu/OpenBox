% plot relu experiment result for MNIST&FMNIST 2 classification
% 2 hidden layers

% TAG
COMPUTE_TEST_SET = true;
PLOT_CONFIG = true;
PLOT_TRANSFORMATION = false;
ANALYZE_EXACT_SAMPLE = false;
ANALYZE_NEURON_REDUNDANT = true;

% compute accuracy on testing data set
if COMPUTE_TEST_SET
    nn = nnff(model.nn, x_test, zeros(size(x_test, 1), model.nn.size(end)));
    py_nn = round(nn.a{end}(:, 2));
    py_lr = h_func([ones(size(x_test, 1), 1), x_test], model.lr);
    fprintf('DNN Test Accuracy = %.4f%%, LR Test Accuracy = %.4f%% \n',...
                100*sum(py_nn==y_test)/size(y_test, 1), 100*sum(round(py_lr)==y_test)/size(y_test, 1));
    figure('name', 'nonneg LR');
    imshow(reshape(model.lr(2:end), 28, 28)); colormap(jet);
end

x_total = [x_sam; x_test];
y_total = [y_sam; y_test];
n_size = size(x_total, 1);
nn = nnff(model.nn, x_total, zeros(n_size, model.nn.size(end)));
py_nn = round(nn.a{end}(:, 2));
fprintf('Total Accuracy = %.4f%% \n', 100*sum(py_nn==y_total)/n_size);

% generate neuralhit 
neuralhit = [];
for i=[2:nn.n-1]
    neuralhit = [neuralhit, (nn.a{i}(:, 2:end))>0];
end
% find unique neuralhits
[neuralhit_unique, ~, loc] = unique(neuralhit, 'rows');
fprintf('In total %d unique neural hits. \n', size(neuralhit_unique, 1));


% ================ COMPUTE CONFIGURATION MATRIX =============
% compute matrix
config_matrix_w = zeros(size(neuralhit_unique, 1), 784);
config_matrix_b = zeros(size(neuralhit_unique, 1), 1);
w_1 = nn.W{1}(:, 2:end); b_1 = nn.W{1}(:, 1); 
w_2 = nn.W{2}(:, 2:end); b_2 = nn.W{2}(:, 1); 
w_output = nn.W{end}(:, 2:end); b_output = nn.W{end}(:, 1);
W={}; B={};
for i = [1:size(neuralhit_unique, 1)]
    % compute w1 and b2
    relu1 = neuralhit_unique(i, 1:size(b_1, 1));
    w_1r=w_1; b_1r = b_1;
    w_1r(relu1==0, :) = 0; b_1r(relu1==0, :) = 0;
    % compute w2 and b2
    relu2 = neuralhit_unique(i, (size(b_1, 1)+1):end);
    w_2r=w_2; b_2r=b_2;
    w_2r(relu2==0, :) = 0; b_2r(relu2==0, :) = 0;
    % compute fw and fb for whole net
    fw = w_output*w_2r*w_1r; fb = w_output*w_2r*b_1r + w_output*b_2r + b_output;
    % plot fw
    W{i}=fw;B{i}=fb;
    
    config_matrix_w(i, :) = fw(1, :)-fw(2, :);
    config_matrix_b(i, :, :) = fb(1)-fb(2);
end

% compute similarity matrix of config_matrix_w
config_matrix = [config_matrix_b config_matrix_w];
similarity_mat = consine_similarity(config_matrix);



% =================================
if PLOT_CONFIG
    % for given x, plot [features and data] of this configuration
    % given x ==> find configuration & group ==> plot samples & features
    % k=3001;
    % % select data in group
    % id = loc(k);    % which group this belongs to
    for id = [1:size(neuralhit_unique, 1)]
        x_group = x_total(loc==id,:);
        y_group = y_total(loc==id, :);
        py_group = py_nn(loc==id, :);
        fprintf('This configuration group: %d = (0: %d, 1:%d)\n', size(x_group, 1), sum(py_group==0), sum(py_group==1));
        % get w and b 
        w_group = squeeze(config_matrix_w(id, :));
        b_group = squeeze(config_matrix_b(id, :));
        % plot features and images
        figure('name',sprintf('%d(%d,%d)', size(x_group, 1), sum(py_group==0), sum(py_group==1))); 
        subplot(2,2,1);     % features for class 0
        shape0 = sum(x_group(py_group==0, :), 1)/sum(py_group==0);
        imshow(reshape(shape0, 28, 28));
        subplot(2, 2, 2);
        shape1 = sum(x_group(py_group==1, :), 1)/sum(py_group==1);
        imshow(reshape(shape1, 28, 28));
        subplot(2, 2, 3); 
    %     imshow(reshape(W{id}(1, :)/max(max(W{id}(1, :))), 28, 28));
        imshow(reshape(abs(W{id}(1, :)), 28, 28));
        subplot(2, 2, 4);
    %     imshow(reshape(W{id}(2, :)/max(max(W{id}(2, :))), 28, 28));
        imshow(reshape(abs(W{id}(2, :)), 28, 28));
    %     imshow(reshape(w_group, 28, 28));
        colormap(jet);
    end
end


% ================ PLOT TRANSFORMATION SPACE ================
if PLOT_TRANSFORMATION
    % get new [x] after DNN transformation 
    x_trans = nn.a{end-1}(:, 2:end);

    % in transferred space, compute decision boundary (for softmax)
    fw=nn.W{end}(:, 2:end); fb=nn.W{end}(:, 1);
    boundary_x1 = [min(x_trans(:, 1))-0.1, max(x_trans(:, 1))+0.1];
    boundary_x2 = ((fw(1,1)-fw(2,1))*boundary_x1+(fb(1)-fb(2)))/(fw(2,2)-fw(1,2));
    % plot x dimension after transformation
    figure; 
    plot(x_trans(py_nn==1, 1), x_trans(py_nn==1, 2), 'r.', ...
            x_trans(py_nn==0, 1), x_trans(py_nn==0, 2), 'b.', boundary_x1, boundary_x2);



    % plot colored transformation space region
    x_plot = x_trans;
    n_size = size(neuralhit, 1);
    a_colored = zeros(size(neuralhit, 1), 1);
    count = 0;
    figure('name', 'transformation space');  
    hold on;
    for k=[1:size(neuralhit, 1)]
        if a_colored(k) ~= 1
            count = count+1;

            sample=neuralhit(k,:);
            a=zeros(n_size, 1);
            for i = [1:n_size]
                if isequal(sample, neuralhit(i, :))
                    a(i)=1;
                end
            end

            x_found = x_plot(a==1, :);
            py_found = py_nn(a==1, :);
            randcolor = rand(1, 3);
            plot(x_found(:, 1), x_found(:, 2), '.', 'Color', randcolor); 
            a_colored = a_colored + a;

            fprintf('k=%d, act-neuron: %d, group size = %d (%d,  %d)\n', k, sum(sample), sum(a), sum(py_found==0), sum(py_found==1));
        end    
    end
    plot(boundary_x1, boundary_x2);
    hold off;    
    fprintf('In total %d regions. \n', count);
end
    

% ==================== EXACT SAMPLE ANALATIC ======================   
if ANALYZE_EXACT_SAMPLE
    % analyze exact sample, its decided features, and after cover features
    kl=[1,100,200,500];
    for k=kl 
        % find configuration
        id = loc(k);
        configuration=neuralhit_unique(id,:);
        % select samples in this configuration group
        thisx = x_total(k, :);
        thisy = py_nn(k);

        % FIRST CASE: decision feature plot
        featured_img{1} = W{id}(1,:).*thisx;
        featured_img{2} = W{id}(2,:).*thisx;
        figure('name', 'case sample presentation');
        subplot(1, 4, 1); imshow(reshape(thisx, 28, 28));
        xlabel(sprintf('[%.4f, %.4f]', nn.a{end}(k, :)));
        subplot(1, 4, 2); imshow(reshape(featured_img{1}*1.3, 28, 28));
        xlabel(sprintf('dpt=%.4f', sum(featured_img{1})));
        subplot(1, 4, 3); imshow(reshape(featured_img{2}*1.3, 28, 28));
        xlabel(sprintf('dpt=%.4f', sum(featured_img{2})));
        fprintf('X id = %d, classified label = %d\n', k, thisy);

        % SECOND CASE: remove decision feature, reclassify
        [~, covered_point] = sort(featured_img{thisy+1});
        covered_point = covered_point(end-20:end);
        thisx(covered_point) = 0;
        subplot(1, 4, 4); imshow(reshape(thisx, 28, 28));
        % compute out = softmax(wx_b)
        out = (W{id}*thisx'+B{id});
        out = exp(out);
        out = out/sum(out);
        xlabel(sprintf('[%.4f, %.4f]', out));
        fprintf('After cover decided features, out = [%.4f, %.4f] \r\n\n', out);
    end
end



% =======================================================
% solve neuron functions, and compute redundant constraint
if size(neuralhit_unique, 1)>10
    a = input('neuralhit size %d, to plot it? 0/1 >>', size(neuralhit_unique, 1));
    ANALYZE_NEURON_REDUNDANT = (a~=0);
end
if ANALYZE_NEURON_REDUNDANT
    w_1 = nn.W{1}(:, 2:end); b_1 = nn.W{1}(:, 1); 
    w_2 = nn.W{2}(:, 2:end); b_2 = nn.W{2}(:, 1); 
    w_output = nn.W{end}(:, 2:end); b_output = nn.W{end}(:, 1);
    bound = [];
    for i = [1:size(neuralhit_unique, 1)]
        % first layer, w is original network's w
        % second layer, w computed from first layer
        if strcmp(nn.activation_function, 'relu')   
            relu1 = neuralhit_unique(i, 1:size(b_1, 1))';
            w2 = w_2*(w_1.*repmat(relu1, 1, size(w_1, 2))); 
            b2 = w_2*(b_1.*relu1)+b_2; 
            w = [w_1; w2];
            b = [b_1; b2];
        elseif strcmp(nn.activation_function, 'relu-nonneg')
            relu1 = neuralhit_unique(i, 1 : size(w_1, 1))';
            w2 = w_2*(w_1.*repmat(relu1, 1, size(w_1, 2)));
            b2 = w_2*((b_1).*relu1)+b_2;
            w = [w_1; w2];
            b = [b_1; b2];
        end


        % if act == 1, -w, b; else w, -b
        relu = neuralhit_unique(i, :);
        w(relu==1, :) = -1*w(relu==1, :);
        b(~relu, :) = -1*b(~relu, :);       % we can plot w to see how these features look like

        thisbound = linear_redundant_constraints(w, b, zeros(size(w, 2), 1), ones(size(w, 2), 1));
        thisbound = thisbound.*(relu-0.5)*2; 
        bound = [bound; thisbound];
    end
end
 
    

    
    
    
    
    