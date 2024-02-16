function rate = LR_Accuracy(x, y, theta)

x_bias = [ones(size(x,1), 1), x];
h = h_func(x_bias, theta);
accur = 0;

for i = 1 : size(x, 1)
   if(abs(h(i) - y(i)) < 0.5)
       accur = accur + 1;
       %check: fprintf('%f -- %f -- acc\n', h(i), y(i));
       %else
       %check: fprintf('%f -- %f\n', h(i), y(i));
   end
end

rate = accur / size(x, 1);