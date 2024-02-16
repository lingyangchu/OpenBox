function svmStruct = svm(x_sam, y_sam, sigma)

% reset y_sam(0, 1) to (-1, 1)
groups = 2*(y_sam-0.5);

% call svm lib
svmStruct = svmtrain(x_sam, groups, 'ShowPlot', true, ...
...%                         'kernel_function', 'rbf', ...
                        'method', 'SMO', ...
                        'boxconstraint',0.01 ...         % C
...%                        'rbf_sigma', sigma ...              % gamma
                        );

% run training data result
py_svm = svmclassify(svmStruct, x_sam);
py_svm = 0.5+(py_svm/2);
fprintf('SVM on training data, accuracy is %.4f\n',...
            sum(py_svm==y_sam)/size(y_sam, 1));


% test
% ptest_svm = svmclassify(svmStruct, x_test);
% ptest_svm = 0.5 + (ptest_svm/2);
% fprintf('SVM on testing data, accuracy is %.4f\n', sum(ptest_svm==y_test)/size(y_test, 1));


