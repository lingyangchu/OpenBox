function Large_Movie_Review()
% this function is to generate input data matrix of movie review dataset


n_feature = 89527;
n_data = 25000; 

data = sparse(n_data, n_feature);
ground_truth = zeros(n_data, 1);
count = 1;
t_start = clock;
f_feat = fopen('aclImdb_v1/labeledBow_test.feat');
while ~feof(f_feat)
    fline = fgetl(f_feat);
    elements = str2double(strsplit(fline, {' ',':'}));
    ground_truth(count) = elements(1);
    cols = elements(2:2:end)+1;
    values = elements(3:2:end);
    data(sub2ind(size(data), ones(1, length(cols))*count, cols)) = values;
    count = count + 1;
    if rem(count, 1000)==0
        fprintf('count = %d ...\n', count);
    end
end
fclose(f_feat);
t_stop=clock;
fprintf('Finish loading, running time %.2f \n', etime(t_stop, t_start));

save('aclImdb_v1/training.mat', 'data','ground_truth');







