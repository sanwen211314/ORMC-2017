% ORMC  % 2017-10-19
% object function min ||X-UV-b1^T||2,1  S.T. U'U = i

function out = ORMC(X,N,r,NITER)

%% Input: 
%   X: data matrix
%   N: observed matrix
%   r : the rank of disered matrix
%   NITER: the number of iteration 
fprintf('Test ORMC')
%%  initialize
[m,n] = size(X);
z = find(N ~= 0);  
nz = find(N == 0);
X(nz) = 0;
data = X(z); % the value of known entries 
d = ones(n,1);   % weighted
[U, S, V] = svds(X,r);
V = S*V';
V = V';
re = zeros(NITER,1);
b = zeros(m,1);
tic
for k = 1:NITER
    
    % update D
    temp_d = sqrt(d);
    D = spdiags(temp_d,0,n,n);
    inD = spdiags(1./temp_d,0,n,n); 
    % initialize
    V_hat = V'*D;
    X_hat = X*D;
    d_s = ones(n,1)'*D; 
   % update b
     b = X_hat * d_s'/(d_s*d_s');           
   % update U 
    temp = X_hat - b*d_s;
    temp1 = temp*V_hat';
    [Us,sigma,Ud] = svd(temp1,'econ'); % stable
    U = Us*Ud';
    % update V_deta
    V_hat = U'*(X_hat - b*d_s);
    % update V
    V = V_hat*inD;
    V = V';
   % update known data
    X = U*V'+ b*ones(1,n);
    X(z)= data;
    % update D
    tempE = X-U*V'-b*ones(1,n);
    Bi = sqrt(sum(tempE.*tempE,1)+eps)';
    d = 0.5./(Bi);
    obj = sum(sqrt(sum(tempE.*tempE,1)));
    if obj<1e-6
        break;
    else
    re(k) = obj;
    end
    if mod(k,50)==0
    display(strcat('In the ',num2str(k), '-th iteration'));
    end
end
t1 = toc;
display(strcat('the time of iteration is£º',num2str(t1),'s'));
out.matrix = X;
out.U = U;
out.b = b;
out.re = re;
[~,index] = sort(Bi);
out.index = index;
out.weight = d;

