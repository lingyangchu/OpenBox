function bound = linear_redundant_constraints(A, b, lb, ub)
% 3 parameters: f, A, b
% optimize min f*x
% s.t.      Ax<=b
%           x1,x2,...xn >=0

n = size(A, 2); 
m = size(A, 1);
z = zeros(m, 1);

for i = [1:m]
    ai = A(i, :);   % ax<=b, we need to max ax
    pi = -ai;       % it equals to min -ax
    A_const = A; A_const(i, :)=[];
    b_const = b; b_const(i)=[];
    x = linprog(pi, A_const, b_const,[],[],lb,ub);
    z(i)=ai*x;
%     fprintf('[%.8f %.8f]\n', ai*x, b(i));
end
redund = (z<=b)';
bound = ~redund; 


