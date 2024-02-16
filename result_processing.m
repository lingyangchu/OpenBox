function result = result_processing(result)

k = result.iter;
result.trainset(k+1) = compute_avg(result.trainset(1:k));
result.validset(k+1) = compute_avg(result.validset(1:k));
result.testset(k+1) = compute_avg(result.testset(1:k));
% result.trainset_pruned(k+1) = compute_avg(result.trainset_pruned(1:k));
% result.validset_pruned(k+1) = compute_avg(result.validset_pruned(1:k));
% result.testset_pruned(k+1) = compute_avg(result.testset_pruned(1:k));




function avg = compute_avg(reslist)
avg = reslist(1);
n = length(reslist);
for i = 2:n
%    avg.dtree = avg.dtree + reslist(i).dtree;
%    avg.svm = avg.svm + reslist(i).svm;
%    avg.adaboost = avg.adaboost + reslist(i).svm;
   avg.lr = avg.lr + reslist(i).lr;
% %    avg.dpmm = avg.dpmm + reslist(i).dpmm;
%    avg.deep_model = avg.deep_model + reslist(i).deep_model;
%    avg.dtlr = avg.dtlr + reslist(i).dtlr;
%    avg.match_deepmodel = avg.match_deepmodel + reslist(i).match_deepmodel;
end
% avg.dtree = avg.dtree/n;
% avg.svm = avg.svm/n;
% avg.adaboost = avg.adaboost/n;
avg.lr = avg.lr/n;
% % avg.dpmm = avg.dpmm/n;
% avg.deep_model = avg.deep_model/n;
% avg.dtlr = avg.dtlr/n;
% avg.match_deepmodel = avg.match_deepmodel/n;







