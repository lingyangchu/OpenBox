function plot_DecisionBoundary(x, y, tree)

% compute amount of leaf nodes
if strcmp(tree.w, 'null')==1
    % leaf node, plot decision boundary
    decisionBoundary(x, y, tree.theta);
else
    % not a leaf node, find child node
    % decisionBoundary(x, y, tree.theta);
    S = Split_Dataset(x, tree.w);
    plot_DecisionBoundary(x(S(:, 1), :), y(S(:, 1)), tree.left);
    plot_DecisionBoundary(x(S(:, 2), :), y(S(:, 2)), tree.right);
end

end


function decisionBoundary(x, y, theta)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

global plot_subid;

% construct a graph area
% construct a new figure for 12 images
if(rem(plot_subid, 12) == 1)
    figure;
    plot_subid = 1;
end
subplot(3,4,plot_subid);
plot_subid = plot_subid + 1;

plot_point(x, y);
hold on

% 2-dimensiion
if size(x, 2) == 2
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(x(:,1))-0.2,  max(x(:,1))+0.2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(1) + theta(2).*plot_x);

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y);
    
    % Legend, specific for the exercise
    %legend('y=0', 'y=1', 'Decision Boundary')
    %axis([-1, 10, -1, 10])
% 3-dimension    imshow((reshape(sum(x0~=0), 28, 28)./4000)>=1)
else
    % pick out top 3 features
    [sorted, indices] = sort(abs(theta(2:size(theta))), 'descend');
    opt_x = [x(:, indices(1)), x(:, indices(2)), x(:, indices(3))];
    opt_theta = [theta(1), theta(indices(1)+1), theta(indices(2)+1), theta(indices(3)+1)];
    
    plot_x = [min(opt_x(:, 1))-2, max(opt_x(:, 1))+2];
    plot_y = [min(opt_x(:, 2))-2, max(opt_x(:, 2))+2];
    plot_z = zeros(2,2);
    plot_z(1,1) = (-1/opt_theta(4))*(opt_theta(1) + opt_theta(2)*plot_x(1) + opt_theta(3)*plot_y(1));
    plot_z(1,2) = (-1/opt_theta(4))*(opt_theta(1) + opt_theta(2)*plot_x(1) + opt_theta(3)*plot_y(2));
    plot_z(2,1) = (-1/opt_theta(4))*(opt_theta(1) + opt_theta(2)*plot_x(2) + opt_theta(3)*plot_y(1));
    plot_z(2,2) = (-1/opt_theta(4))*(opt_theta(1) + opt_theta(2)*plot_x(2) + opt_theta(3)*plot_y(2));
    surf(plot_x, plot_y, plot_z);
    xlabel(strcat('attr-',num2str(indices(1))));
    ylabel(strcat('attr-',num2str(indices(2))));
    zlabel(strcat('attr-',num2str(indices(3))));
%     alpha(.2);
    
%     % Here is the grid range
%     u = linspace(-1, 1.5, 50);
%     v = linspace(-1, 1.5, 50);
% 
%     z = zeros(length(u), length(v));
%     % Evaluate z = theta*x over the grid
%     for i = 1:length(u)
%         for j = 1:length(v)
%             z(i,j) = mapFeature(u(i), v(j))*theta;
%         end
%     end
%     z = z'; % important to transpose z before calling contour
% 
%     % Plot z = 0
%     % Notice you need to specify the range [0, 0]
%     contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end


% function subplot_point(x, y)
% % plot the points x , different color means different value of y
% % x_dimension = 2
% % y = {0, 1}
% 
% D0 = [];
% D1 = [];
% 
% for i = 1 : size(x, 1)
%    if(y(i) == 0)
%        D0 = [D0; x(i, :)];
%    else
%        D1 = [D1; x(i, :)];
%    end
% end
% 
% % plot random numbers
% if length(D0)==0
%     plot(D1(:,1), D1(:,2), 'b+');
% elseif length(D1)==0
%     plot(D0(:,1), D0(:,2), 'r+');
% else
%     plot(D0(:,1), D0(:,2), 'r+', D1(:,1), D1(:,2), 'b+');
% end
% 
% end