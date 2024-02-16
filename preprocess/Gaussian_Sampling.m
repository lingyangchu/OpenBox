function [x_sam,y_sam] = Gaussian_Sampling(n, num_sample)
%   y=0     y=1     | y = 1
%   y=1     y=0     |

x_sam = [];
y_sam = [];
num_point = n/num_sample;

% 4 samples
if num_sample == 4 || num_sample == 6
    a = 0.1;
    b = 0.7;
    c = 1.5;
    mu = [b,a; a,b; c,b; b,c];  % diamond
%     mu = [a,a; a,b; b,a; b,b];
%     mu = [a,a; a,b; b,a; b,b; c,a; c,b];
    sigma = [0.05, 0; 0, 0.05];

    rng default
    for i = 1 : num_sample
       r = mvnrnd(mu(i, :), sigma, num_point); 
       x_sam = [x_sam;r];
       if(i == 1 || i == 4 || i == 5)
          y_sam = [y_sam; ones(num_point, 1)]; 
       else
          y_sam = [y_sam; zeros(num_point, 1)];
       end
    end
    % % use distance to control and divide classes
%     dist_1 = max(...
%                     [((x_sam(:, 1) - mu(1, 1)).^2 + (x_sam(:, 2)-mu(1, 2)).^2), ...
%                     ((x_sam(:, 1)-mu(4, 1)).^2 + (x_sam(:, 2)-mu(4, 2)).^2)]' ...
%                 )';
%     dist_0 = max(...
%                     [((x_sam(:, 1)-mu(2, 1)).^2 + (x_sam(:, 2)-mu(2, 2)).^2), ...
%                     ((x_sam(:, 1)-mu(3, 1)).^2 + (x_sam(:, 2)-mu(3, 2)).^2)]' ...
%                 )';
%     y_sam = [dist_1 > dist_0];
elseif num_sample == 5
    a = 0.1;
    b = 1;
    c = 2;
    mu = [a,a; a,b; b,a; b,b; c,a];
    sigma = [0.05, 0; 0, 0.05];
    
    rng default;
    for i = 1 : num_sample
        r = mvnrnd(mu(i, :), sigma, num_point);
        x_sam = [x_sam; r];
        if(i == 1 ||i == 4 || i == 5)
            y_sam = [y_sam; ones(num_point, 1)];
        else
            y_sam = [y_sam; zeros(num_point, 1)];
        end
    end
end

% plot random numbers
figure;
plot(x_sam(y_sam==1,1), x_sam(y_sam==1, 2), 'x', x_sam(y_sam==0, 1), x_sam(y_sam==0, 2), '+');




