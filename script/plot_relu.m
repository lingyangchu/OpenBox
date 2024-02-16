% 2-D relu plot. including: decision boundary, region-divided plot, x-after transformation plot

%% data: circle

x_total = [x_sam; x_test];
y_total = [y_sam; y_test];

% 1. plot the whole decision boundary image
figure('name', 'basic');    % plot synthetic data
plot(x_total(y_total==0, 1), x_total(y_total==0, 2), 'r.', ...
        x_total(y_total==1, 1), x_total(y_total==1, 2), 'b.');
n_size = size(x_total, 1);  % plot dnn boundary
nn = nnff(model.nn, x_total, zeros(n_size, model.nn.size(end)));
py_nn = nn.a{end}(:, 2);
figure; scatter(x_total(:, 1), x_total(:, 2), [], round(py_nn, 1));
py_nn = round(py_nn);
figure('name', 'dnn');
plot(x_total(py_nn==0, 1), x_total(py_nn==0, 2), 'r.', ...
        x_total(py_nn==1, 1), x_total(py_nn==1, 2), 'b.');

% generate neuralhit from debug
neuralhit = [];
for i=[2:nn.n-1]
    neuralhit = [neuralhit, (nn.a{i}(:, 2:end))>0];
end

% get new x' after DNN transformation 
x_trans = nn.a{end-1}(:, 2:end);

%       compute decision boundary for sigmoid
% boundary_x1 = [min(x_trans(:, 1)), max(x_trans(:, 1))];
% fw=nn.W{end}(2, 2:end); fb=nn.W{end}(2, 1);
% boundary_x2 = (fb+fw(1)*boundary_x1)/(-1*fw(2));
%       compute decision boundary for softmax
boundary_x1 = [min(x_trans(:, 1))-0.1, max(x_trans(:, 1))+0.1];
fw=nn.W{end}(:, 2:end); fb=nn.W{end}(:, 1);
boundary_x2 = ((fw(1,1)-fw(2,1))*boundary_x1+(fb(1)-fb(2)))/(fw(2,2)-fw(1,2));
% plot x dimension after transformation
figure; 
plot(x_trans(py_nn==1, 1), x_trans(py_nn==1, 2), 'r.', ...
        x_trans(py_nn==0, 1), x_trans(py_nn==0, 2), 'b.', boundary_x1, boundary_x2);

% find x on same decision boundary
% if plot real space x
x_plot = x_total;
% if plot transformation space x'
% x_plot = x_trans;
n_size = size(neuralhit, 1);
a_colored = zeros(size(neuralhit, 1), 1);
figure('name', 'dnn');  
hold on;
for k=[1:size(neuralhit, 1)]
    if a_colored(k) ~= 1

        sample=neuralhit(k,:);
        a=zeros(n_size, 1);
        for i = [1:n_size]
            if isequal(sample, neuralhit(i, :))
                a(i)=1;
            end
        end

        x_found = x_plot(a==1, :);
        net = nnff(model.nn, x_found, zeros(size(x_found, 1), model.nn.size(end)));
        y_found = net.a{end}(:, 2); py_nn=round(y_found);
        randcolor = rand(1, 3);
        plot(x_found(:, 1), x_found(:, 2), '.', 'Color', randcolor);
%         plot(x_found(py_nn==0, 1), x_found(py_nn==0, 2), 'r.', ...
%                 x_found(py_nn==1, 1), x_found(py_nn==1, 2), 'b.');
%         axis([-1.5, 1.5, -1.5, 1.5]);   
        a_colored = a_colored + a;
    end    
end
plot(boundary_x1, boundary_x2);
hold off;   

% plot sub-region and their decision boundary
for k=[1:1:20]
        sample=neuralhit(k,:);
        a=zeros(n_size, 1);
        for i = [1:n_size]
            if isequal(sample, neuralhit(i, :))
                a(i)=1;
            end
        end
        % compute f: y=f(x), plot decision boundary
        net=model.nn;
%         fw=nn.W{4}(:, 2:end);fb=net.W{4}(:,1);      % for x_after transformation
        w1=net.W{1}(:, 2:end);b1=net.W{1}(:, 1); relu1=sample(1:4);w1(relu1==0, :)=0;b1(relu1==0, :)=0;
        w2=net.W{2}(:, 2:end);b2=net.W{2}(:, 1); relu2=sample(5:36);w2(relu2==0, :)=0;b2(relu2==0,:)=0;
        w3=net.W{3}(:, 2:end);b3=net.W{3}(:, 1); relu3=sample(37:end);w3(relu3==0,:)=0;b3(relu3==0, :)=0;
        w4=net.W{4}(:, 2:end);b4=net.W{4}(:, 1);
        fw=w4*w3*w2*w1; fb=w4*w3*w2*b1+w4*w3*b2+w4*b3+b4;
        
        x_found = x_plot(a==1, :);
        y_found = py_nn(a==1);
%         if sum(py_nn)~=size(py_nn, 1) & sum(py_nn)~=0
            figure; hold on;
            plot(x_found(y_found==0, 1), x_found(y_found==0, 2), 'r.', ...
                    x_found(y_found==1, 1), x_found(y_found==1, 2), 'b.');
            x=[0,120]; plot(x, (fb(2)+fw(2,1)*x)/(-1*fw(2,2)));
            axis([0,120,0,120]);     
            hold off;
%         end
end

    