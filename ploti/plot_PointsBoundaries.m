function plot_PointsBoundaries(x, y, tree)

% plot points
% plot_point(x, y);
% hold on

% attribute number <= 2
if size(x, 2) < 3
    plot_NodeBoundary(x, y, tree);       
    % Legend, specific for the exercise
    legend('y=0', 'y=1', 'Sub-space Boundary')
end

% hold off
end

function plot_NodeBoundary(x_sam, y_sam, tree)
% plot nothing if this is a leaf node
if strcmp(tree, 'null') | strcmp(tree.w, 'null') | size(x_sam, 1) == 0
    return
end

global plot_subid;

% construct a graph area
% construct a new figure for 12 images
if(rem(plot_subid, 12) == 1)
    figure;
    plot_subid = 1;
end
subplot(3,4,plot_subid);
plot_subid = plot_subid + 1;

plot_point(x_sam, y_sam);
hold on;

w0 = tree.w(1);
w1 = tree.w(2);
w2 = tree.w(3);
x_range = [min(x_sam(:, 1)), max(x_sam(:, 1))];
y_range = (-1/w2)*(w0+w1*x_range);
plot(x_range, y_range);

hold off;

% % plot boundary of the data area
% if tree.attrId == 1
%     % plot boundary of each range
%     range = [min(x_sam(:, 2)), max(x_sam(:, 2))];
%     plot([tree.split, tree.split], range);    
% else
%     range = [min(x_sam(:, 1)), max(x_sam(:, 1))];
%     plot(range, [tree.split, tree.split]);
% end


S = Split_Dataset(x_sam, tree.w);
plot_NodeBoundary(x_sam(S(:, 1), :), y_sam(S(:, 1)), tree.left);
plot_NodeBoundary(x_sam(S(:, 2), :), y_sam(S(:, 2)), tree.right);

end




