function [py_cnn, net] = CNN(x_sam, y_sam, tag)

if strcmp(tag, 'cifar10')
    [py_cnn, net] = CNN_cifar10(x_sam, y_sam);
elseif strcmp(tag, 'fashionmnist')
    [py_cnn, net] = CNN_fmnist(x_sam, y_sam);
elseif strncmp(tag, 'mnist', 5)
    [py_cnn, net] = CNN_mnist(x_sam, y_sam, tag);
elseif strcmp(tag, 'ck15')
    [py_cnn, net] = CNN_ckplus(x_sam, y_sam); 
end



