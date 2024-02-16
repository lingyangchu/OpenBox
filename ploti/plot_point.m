function plot_point(x, y)
% plot the points x , different color means different value of y
% x_dimension = 2
% y = {0, 1}

if size(x, 2) == 2
    plot_2dpoint(x, y);
else
    plot_3dpoint(x, y);
end
end

function plot_2dpoint(x, y)

id1 = (y == 1);
id0 = ~id1;

if sum(id1) == 0
    plot(x(id0, 1), x(id0, 2), 'b.');
elseif sum(id0) == 0
    plot(x(id1, 1), x(id1, 2), 'r+');
else
    plot(x(id0, 1), x(id0, 2), 'b.', x(id1, 1), x(id1, 2), 'r+');
end

end

function plot_3dpoint(x, y)
D0 = [];
D1 = [];

for i = 1 : size(x, 1)
   if(y(i) < 0.5)
       D0 = [D0; x(i, :)];
   else
       D1 = [D1; x(i, :)];
   end
end

% plot random numbers
for i = 1:3:size(x, 2)-2
    if isempty(D0)
        plot3(D1(:,i), D1(:,i+1), D1(:,i+2), 'b+');
    elseif isempty(D1)
        plot3(D0(:,i), D0(:,i+1), D0(:,i+2), 'r+');
    else
        plot3(D0(:,i), D0(:,i+1), D0(:,i+2),'r+', D1(:,i), D1(:,i+1), D1(:,i+2), 'b+');
    end
end

end




