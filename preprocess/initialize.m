function init = initialize()

init.dataset = 'moviereview';       % 'gaussian', 'sine', 'circle', 'animalshape'
                                % 'breastcc', 'spamemail', 'mnist56',  'multifmnist'
                                % 'mnist49', 'mnist17', 'fashionmnist', 'ck15', 'cifar10'
                                % 'moviereview'
                                
init.dl_model = 'dnn';          % deep learning model. 'cnn' or 'dnn'


% init.lamda_theta = 5e-5;        % lamda in L2 norm = 1; lamda in L1 norm = 0.1
% init.lamda_w = 1e-6;
% init.lr = 0.01;                  % learning rate
% init.grad_j_threshold = 1e-10;  % convergence threshold of gradient descent
% init.w_theta_alteroptimize = 0.0005;

global plot_count;
global plot_subid;
% count to plot tree path attribute image
plot_count = 1;     
% count to plot decision boundary
plot_subid = 1;


