
function [x_train, y_train, x_test, y_test] = load_Data(name)
%% in this function, it
%%  (1) generate data or load data from file
%%  (2) convert y to be 0/1 vector
%%  (3) normalize the data
%%  (4) split data set into 2 part, training set and testing set
%% name = {'gaussian', 'breastcc', 'spamemail', 'mnist', 'cifar10'}

if strcmp(name, 'gaussian') == 1
    num_sample = 4;
    num_instances = 2000;
    [x_train, y_train] = Gaussian_Sampling(num_instances, num_sample);
%     x_test = x_train; y_test = y_train;
    [x_test, y_test] = Gaussian_Sampling(num_instances, num_sample);
elseif strcmp(name, 'sine') == 1
    [x_sam, y_sam] = Sine_Sampling();
    x_train = x_sam; 
    y_train = y_sam; 
    
    x_test = [4*pi*rand(10000, 1), 4*rand(10000, 1)-2]; % uniform
    y_test = x_test(:, 2) - sin(x_test(:, 1));
    y_test = (y_test >= 0);
 
elseif strcmp(name, 'circle') == 1
    [x_train, y_train, x_test, y_test] = Circle_Sampling();

elseif strcmp(name, 'animalshape')
    img = rgb2gray(imread('preprocess/animal_shape_1.jpg'));
%     img = imresize(img, 0.5, 'bilinear');
%     n_size = size(img,1)*size(img,2);
%     img = reshape(img, [n_size 1]);
%     x_train = zeros(n_size, 2);
%     y_train = zeros(n_size, 1);
    x_train = []; 
    y_train = [];
    for i = 1:size(img, 1)
        for j = 1:size(img, 2)
             if rem(i, 2)==0 & rem(j, 2) == 0
%                 id = (i-1)*size(img, 2) + j;
%                 x_train(id, :) = [i j];
                x_train = [x_train; i, j];
                if img(i, j) > 128
%                     y_train(id) = 1;
                    y_train = [y_train; 1];
                else
%                     y_train(id) = 0;
                    y_train = [y_train; 0];
                end
             end
        end
    end
    x_train = x_train/256;
    x_test=[];y_test=[];
    
    figure;
    plot(x_train(y_train==1, 1), x_train(y_train==1, 2), 'r.', x_train(y_train==0, 1), x_train(y_train==0, 2), 'b.');
elseif strcmp(name, 'breastcc') == 1
    % read from file
    data = importdata('breast-cancer-wisconsin.data');
    x_sam = data(:, 2:10);  % 680 instances, 430 for training and 150 for testing
    y_sam = data(:, 11);    % class = 2 or 4    
    y_sam = y_sam./2 - 1;   % 0: class = 2; 1: class = 4
    % do normolization - z-score
    for i = 1:size(x_sam, 2)
       avrg = mean(x_sam(:, i));
       deriation = (var(x_sam(:, i))).^(1/2);
       x_sam(:, i) = (x_sam(:, i) - avrg)/deriation;
    end
    % divide into training set and testing set
    x_train = x_sam(1:530, :); x_test = x_sam(531:680, :);
    y_train = y_sam(1:530, :); y_test = y_sam(531:680, :);
    
elseif strcmp(name, 'spamemail') == 1
    % import data from file, do randperm then save data again
%     data = importdata('spambase.data');
%     data_rand = data(randperm(size(data, 1)), :);
%     save('preprocess/spambase_rand.data', 'data_rand');
    data_rand = importdata('spambase_rand.data');
    x_sam = data_rand(:, 1:57);  % 4601 instances, 4000 for training and 601 for testing
    y_sam = data_rand(:, 58);    % class 0: not a spam email; 1: a spam email
    % remove stop words: (make, all, our, over, will, you, your)
    x_sam(:, [1 3 5 6 12 19 21]) = [];
    % do normolization - z-score
%     for i = 1:size(x_sam, 2)
%        avrg = mean(x_sam(:, i));
%        deriation = (var(x_sam(:, i))).^.5;
%        x_sam(:, i) = (x_sam(:, i) - avrg)/deriation;
%     end
    % do normolization - max-min
    for i = 1:size(x_sam, 2)
       max_x = max(x_sam(:, i));
       min_x = min(x_sam(:, i));
       x_sam(:, i) = (x_sam(:, i) - min_x)/(max_x - min_x);
    end
    
    % divide into training set and testing set
    x_train = x_sam(1:3500, :); x_test = x_sam(3501:4600, :);
    y_train = y_sam(1:3500, :); y_test = y_sam(3501:4600, :);

elseif strcmp(name, 'mnist56') == 1
    x_sam = [loadMNISTImages('train-images.idx3-ubyte'), loadMNISTImages('t10k-images.idx3-ubyte')];
    y_sam = [loadMNISTLabels('train-labels.idx1-ubyte'); loadMNISTLabels('t10k-labels.idx1-ubyte')];
    
    % filter & pick up 5&6
    selected_id = (y_sam==5 | y_sam==6);
    x_sam = x_sam(:, selected_id);
    y_sam = y_sam(selected_id);
    y_sam = y_sam-5;    % 0/1 classification
    x_sam = x_sam';
    
    % pick 13000 instances
    x_train = x_sam(1:8000, :);
    y_train = y_sam(1:8000);
    x_test = x_sam(8001:13000, :);
    y_test = y_sam(8001:13000);

elseif strcmp(name, 'mnist49') == 1
    x_sam = [loadMNISTImages('train-images.idx3-ubyte'), loadMNISTImages('t10k-images.idx3-ubyte')];
    y_sam = [loadMNISTLabels('train-labels.idx1-ubyte'); loadMNISTLabels('t10k-labels.idx1-ubyte')];
    
    % filter & pick up 4&9
    selected_id = (y_sam == 4 | y_sam == 9);
    x_sam = x_sam(:, selected_id);
    y_sam = y_sam(selected_id);
    y_sam = (y_sam - 4)/5;      % 0/1 classification, 0is4, 1is9
    x_sam = x_sam';
    
    % pick 11000 instances
    x_train = x_sam(1:8000, :);
    y_train = y_sam(1:8000);  
    x_test = x_sam(8001:13000, :);
    y_test = y_sam(8001:13000);
    
elseif strcmp(name, 'mnist17') == 1
    x_sam = [loadMNISTImages('train-images.idx3-ubyte'), loadMNISTImages('t10k-images.idx3-ubyte')];
    y_sam = [loadMNISTLabels('train-labels.idx1-ubyte'); loadMNISTLabels('t10k-labels.idx1-ubyte')];
     
    % filter & pick up 1 & 7: #=15170
    selected_id = (y_sam == 1 | y_sam == 7);
    x_sam = x_sam(:, selected_id);
    y_sam = y_sam(selected_id);
    y_sam = (y_sam - 1)/6;  % 0/1 classification
    x_sam = x_sam';
    
    %pick 13000 instances
    x_train = x_sam(1:8000, :);
    y_train = y_sam(1:8000);
    x_test = x_sam(8001:13000, :);
    y_test = y_sam(8001:13000);

elseif strcmp(name, 'fashionmnist')
    x_sam = [loadMNISTImages('fashion_mnist/train-images-idx3-ubyte'), loadMNISTImages('fashion_mnist/t10k-images-idx3-ubyte')];
    y_sam = [loadMNISTLabels('fashion_mnist/train-labels-idx1-ubyte'); loadMNISTLabels('fashion_mnist/t10k-labels-idx1-ubyte')];
    
    % chos1: 7 sneaker & 9 ankle boot
    % chos2: 2 pullover & 4 coat
    selected_id = (y_sam == 0 | y_sam == 1);
    x_sam = x_sam(:, selected_id)';
    y_sam = (y_sam(selected_id))/1;     % 0/1
    
    % pick 14000 instances
    x_train = x_sam(1:8000, :);
    y_train = y_sam(1:8000);
    x_test = x_sam(8001:14000, :);
    y_test = y_sam(8001:14000, :);

elseif strcmp(name, 'multifmnist')
    x_sam = [loadMNISTImages('fashion_mnist/train-images-idx3-ubyte'), loadMNISTImages('fashion_mnist/t10k-images-idx3-ubyte')];
    y_sam = [loadMNISTLabels('fashion_mnist/train-labels-idx1-ubyte'); loadMNISTLabels('fashion_mnist/t10k-labels-idx1-ubyte')];

    % choose subset 
    selected_set = [5 7 9];
    selected_id = (y_sam == selected_set(1) | y_sam == selected_set(2) | ...
                    y_sam == selected_set(3));
    x_sam = x_sam(:, selected_id)';
    y_sam = y_sam(selected_id);
    
    x_train = x_sam(1:4000*length(selected_set), :);
    y_train = y_sam(1:4000*length(selected_set));
    x_test = x_sam(4000*length(selected_set)+1:end, :);
    y_test = y_sam(4000*length(selected_set)+1:end, :);
    
elseif strcmp(name, 'moviereview')
    load aclImdb_v1/training.mat;
    x_train = data;
    y_train = ground_truth;
    y_train(y_train<=5) = 0;
    y_train(y_train>5) = 1;

    x_test=[]; y_test = [];
    
elseif strcmp(name, 'ck15')     % anger & happy
    load ck_image_flat.mat;   
    x_sam = imagedata_rand/255;
    y_sam = imagelabel_rand;
    
    x_train = x_sam(1:90, :); x_test = x_sam(91:114, :);
    y_train = y_sam(1:90); y_test = y_sam(91:114);
   
elseif strcmp(name, 'cifar10')      % cat and dog
    load cifar_catdog.mat;
    x_sam = x_gray;
    y_sam = y;       % 0=cat, 1=dog
    
    % minus mean
    dataMean = mean(x_sam, 1);       % data - mean
    x_sam = single(bsxfun(@minus, x_sam, dataMean));
    
    x_train = x_sam(1:8000, :); x_test = x_sam(8001:12000, :);
    y_train = y_sam(1:8000, :); y_test = y_sam(8001:12000);
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% CK+ DATASET %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function readAndSaveCKPlus() 
mFiles = cell(0,0);
% read image and label
% scan and list all files
path_list = {'./ck_1/'; './ck_5/'};
for p = 1:2
    mPath{1} = path_list{p};
    [r, c] = size(mPath);
    while c~= 0
        strPath = mPath{1};
        Files = dir(fullfile(strPath, '*.*'));
        LengthFiles = length(Files);
        if LengthFiles == 0
            break;
        end
        mPath(1) = [];
        iCount = 1;
        while LengthFiles > 0
            if Files(iCount).isdir == 1
                if Files(iCount).name ~= '.'
                    filePath = [strPath Files(iCount).name '/'];
                    [r,c] = size(mPath);
                    mPath{c+1} = filePath;
                end
            else
                filePath = [strPath Files(iCount).name];
                [row, col] = size(mFiles);
                mFiles{col+1} = filePath;
            end

            LengthFiles = LengthFiles - 1;
            iCount = iCount + 1;
        end
        [r, c] = size(mPath); 
    end
end

% read images and save 
imagelabel = zeros(length(mFiles), 1);
imagedata = zeros(length(mFiles), 96*96);
imagelabel(46:length(mFiles)) = 1;       % emotion 5: happy
for i = 1:length(mFiles)
    img = rgb2gray(imread(mFiles{i}));
%     img = imresize(img, 0.5, 'bilinear');
    imagedata(i, :) = reshape(img(:, :, 1), [1, 96*96]);
end
% % randperm
imagelabel_rand = zeros(size(imagelabel));
imagedata_rand = zeros(size(imagedata));
id_rand = randperm(length(mFiles));
for i = 1:length(id_rand)
    id = id_rand(i);
    fprintf('%d = %d\n', i, id);
    imagelabel_rand(i) = imagelabel(id);
    imagedata_rand(i, :) = imagedata(id, :);
end

save('ck_image_flat.mat', 'imagelabel_rand', 'imagedata_rand');
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% CIFAR 10 DATASET %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function readAndSaveCifar10()
% load cifar 10: dog and cat
cd V:\huxiah\research_01\dataset\cifar-10-matlab\cifar-10-batches-mat
batchs = {'data_batch_1'; 'data_batch_2';'data_batch_3';'data_batch_4';'data_batch_5'; 'test_batch'};
x = [];
y = [];     % 0~9
for i = 1:6
    load(batchs{i});
    id_selected = (labels == 3) | (labels == 5);
    x = [x; data(id_selected, :)];
    y = [y; labels(id_selected, :)];
end
x = double(x)/256;
y = double((y-3)/2);    % 0: cat; 1:dog

% convert to gray image
x_gray = zeros(size(x, 1), 32*32);
for i = 1:size(x, 1)
    img = reshape(x(i, :), [32, 32, 3]);
    imggray = rgb2gray(img);
    x_gray(i, :) = reshape(imggray, [1, 32*32]);
end

save('C:\Users\huxiah\Documents\MATLAB\res_nn_01\preprocess\cifar_catdog.mat', 'x_gray', 'y');
clear all;
cd C:\Users\huxiah\Documents\MATLAB\res_nn_01;







