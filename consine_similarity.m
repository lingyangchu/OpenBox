function similarity=consine_similarity(feature_matrix)

n = size(feature_matrix, 1);
distance = feature_matrix*feature_matrix';
% l2-norm
norm_vector = zeros(n, 1);
for i = [1:n]
    norm_vector(i) = norm(feature_matrix(i, :), 2);
end
norm_matrix = norm_vector*norm_vector';

similarity = distance./norm_matrix;
