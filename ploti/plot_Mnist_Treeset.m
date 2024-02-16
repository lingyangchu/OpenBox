function plot_Mnist_Treeset(tree, x_train, y_train)
%%%%%%%%%%%%% 0/5 display at each node
%%%%%%%%%%%%% to see how 0/5 looks like at each subset

global plot_count;
if rem(plot_count, 12) == 1
    figure;
    plot_count = 1;
end

% plot 0 at this node
subplot(3, 4, plot_count);
zero_plot = x_train(y_train==0, :);
if size(zero_plot, 1) ~= 1
    zero_plot = sum(zero_plot);
end
zero_plot = zero_plot/max(zero_plot);
imshow(reshape(zero_plot, 28, 28));
xlabel(strcat('five no.', num2str(length(y_train))));
plot_count = plot_count + 1;

% plot 5 at this node
subplot(3, 4, plot_count);
five_plot = x_train(y_train==1, :);
if size(five_plot, 1) ~= 1
    five_plot = sum(five_plot);
end
five_plot = five_plot/max(five_plot);
imshow(reshape(five_plot, 28, 28));
xlabel(strcat('six no.', num2str(length(y_train))));
plot_count = plot_count + 1;

% plot 0 - 5
subplot(3, 4, plot_count);
% if length(five_plot) ~= 0 & length(zero_plot) ~= 0
rem0_plot = zero_plot - five_plot;
rem0_plot(rem0_plot < 0) = 0;
rem0_plot = rem0_plot/max(rem0_plot);
imshow(reshape(rem0_plot, 28, 28));
xlabel(strcat('0 remains no.', num2str(length(y_train))));
plot_count = plot_count + 1;

% plot 5 - 0
subplot(3, 4, plot_count);
% if length(five_plot) ~= 0 & length(zero_plot) ~= 0
rem1_plot = five_plot - zero_plot;
rem1_plot(rem1_plot < 0) = 0;
rem1_plot = rem1_plot/max(rem1_plot);
imshow(reshape(rem1_plot, 28, 28));
xlabel(strcat('1 remains no.', num2str(length(y_train))));
plot_count = plot_count + 1;

% % plot 0 xor 5
% subplot(3, 4, plot_count);
% % if length(five_plot) ~= 0 & length(zero_plot) ~= 0
% xor_plot = five_plot + zero_plot;
% xor_plot(((five_plot > 0.3).*(zero_plot > 0.3)) ~= 0) = 0;
% xor_plot = xor_plot/max(xor_plot);
% % xor_plot = rem0_plot + rem1_plot;
% % xor_plot = xor_plot/max(xor_plot);
% imshow(reshape(xor_plot, 28, 28));
% xlabel(strcat('xor no.', num2str(length(y_train))));
% plot_count = plot_count + 1;
% 
% % plot 0 union 5
% subplot(3, 4, plot_count);
% % if length(five_plot) ~= 0 & length(zero_plot) ~= 0
% union_plot = (five_plot + zero_plot)/2;
% % union_plot = union_plot*2/max(union_plot);
% imshow(reshape(union_plot, 28, 28));
% xlabel(strcat('union no.', num2str(length(y_train))));
% plot_count = plot_count + 1;

% if this is not a leaf node
if strcmp(tree.w, 'null') == 0
    S = Split_Dataset(x_train, tree.w); 
%     % % add a new view, left_set xor right_set
%     subplot(3, 4, plot_count);
%     plot_l = sum(x_train(S(:, 1), :));
%     plot_r = sum(x_train(S(:, 2), :));
%     plot_l = plot_l*2/max(plot_l);
%     plot_r = plot_r*2/max(plot_r);
%     
%     xor_plot = plot_l;
%     xor_plot((xor_plot < 0.1) ~= 0) = 0;
%     xor_plot(((plot_l > 0.5).*(plot_r > 0.5)) ~= 0) = 0;
% %     xor_plot((plot_r > 0.6) ~= 0) = 0;
%     
%     xor_plot = xor_plot*3/max(xor_plot);
%     imshow(reshape(xor_plot, 28, 28));
%     xlabel(strcat('xor_child no.', num2str(length(y_train))));
%     plot_count = plot_count + 1;
    
    plot_Mnist_Treeset(tree.left, x_train(S(:, 1), :), y_train(S(:, 1)));
    plot_Mnist_Treeset(tree.right, x_train(S(:, 2), :), y_train(S(:, 2)));
end

