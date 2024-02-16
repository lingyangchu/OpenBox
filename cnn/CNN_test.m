function py_cnn = CNN_test(net, x_sam, y_sam, tag)

if strcmp(tag, 'cifar10')
    [py_cnn, net] = CNN_cifar10(x_sam, y_sam);
elseif strcmp(tag, 'fashionmnist')
    x = permute(x_sam, [2 1]);
    x = reshape(x, [28 28 1 size(x_sam, 1)]);
    imdb.images.data = single(x);
    res = vl_simplenn(net, imdb.images.data);
    py_cnn = squeeze(gather(res(end).x));
    py_cnn = py_cnn(2, :)';
elseif strncmp(tag, 'mnist', 5)
    [py_cnn, net] = CNN_mnist(x_sam, y_sam, tag);
elseif strcmp(tag, 'ck15')
    [py_cnn, net] = CNN_ckplus(x_sam, y_sam); 
end



