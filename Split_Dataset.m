function S = Split_Dataset(x, w)
% this is a binary tree with 2 children 

h_w = h_func([ones(size(x, 1), 1), x], w);
S = [round(h_w)==0, round(h_w)==1];



% METHOD 3: DECISION TREE MAX-LIKELIHOOD MODEL
% S = [x_sam(:, attrId) < split, x_sam(:, attrId) >= split];


