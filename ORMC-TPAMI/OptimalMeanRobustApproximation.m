function [M_est,U,V,b] = OptimalMeanRobustApproximation(M,W,r,lambda,rho,maxIterIN)
%% Robust low-rank matrix approximation with missing data and outliers
% min |W.*(M-E)|_1 + lambda*|V|_*
% s.t., E = UV+be', U'*U = I
%
%Input: 
%   M: m*n data matrix
%   W: m*n indicator matrix, with '1' means 'observed', and '0' 'missing'.
%   r: the rank of r
%   lambda: the weighting factor of the trace-norm regularization, 1e-3 in default.
%   rho: increasing ratio of the penalty parameter mu, usually 1.05.
%   maxInterIN: maximum iteration number of inner loop, usually 100.
%   signM: if M>= 0, then signM = 1, otherwise, signM = 0;
%Output:
%   M_est: m*n full matrix, such that M_est = U_est*V_est
%   U_est: m*r matrix
%   V_est: r*n matrix
%   L1_error: the L1-norm error of observed data only.
%% In-default parameters
[m n] = size(M); %matrix dimension
if nargin < 6
    maxIterIN = 100;
end
if nargin < 5
    rho = 1.05;
end
if nargin < 4
    lambda = 1e-3;
end
if nargin < 3
    disp('Please input the data matrix M, the indicator W and the rank r, and try again.');
end

maxIterOUT = 500;
max_mu = 1e20;
mu = 1e-6;
M_norm = norm(M,'fro');
tol = 1e-8*M_norm;

cW = ones(size(W)) - W; %the complement of W.
display = 1; %display progress
%% Initializing optimization variables as zeros
E = zeros(m,n);
U = zeros(m,r);
V = zeros(r,n);
Y = zeros(m,n); %lagrange multiplier
%% Start main outer loop
iter_OUT = 0;
objs=[];
In = [];
b = zeros(m,1);
while iter_OUT < maxIterOUT
    iter_OUT = iter_OUT + 1;
    
    itr_IN = 0;
    obj_pre = 1e20;
    %start inner loop
    while itr_IN < maxIterIN 
        %update U
        tb = repmat(b,[1,n]);
        %    b*ones(1,n);
        temp = (E -tb+ Y/mu)*V';
        
        % update U by QR
        [U,~] = qr(temp,0); 
        
        % update V
        V = (mu/(mu+lambda))*U'*(E - tb + Y/mu);
        sigma0 = 0;
        UV = U*V;
       % update b
        b = (1/n)*(E-UV + Y/mu)*ones(n,1);
       % update E
        temp1 = UV + repmat(b,[1,n]) - Y/mu;
        temp = M-temp1;
        E = max(0,temp - 1/mu) + min(0,temp + 1/mu);    
        E = (M-E).*W + temp1.*cW;        
        
        %evaluate current objective
        obj_cur = sum(sum(abs(W.*(M-E)))) + lambda*norm(V,'fro')^2 + sum(sum(Y.*(E-UV))) + mu/2*norm(E-UV,'fro')^2;

        %check convergence of inner loop
        if abs(obj_cur - obj_pre) <= 1e-8*abs(obj_pre)
            break;
        else
            obj_pre = obj_cur;
            itr_IN = itr_IN + 1;
        end
    end

    leq = E - temp1- Y/mu;
    
    %%
    stopC = norm(leq,'fro');
    if display
        obj = sum(sum(abs(W.*(M-UV)))) + lambda*norm(V,'fro')^2;
        objs = [objs,obj];
    end
    if display && (iter_OUT==1 || mod(iter_OUT,50)==0 || stopC<tol)
        disp(['iter ' num2str(iter_OUT) ',mu=' num2str(mu,'%2.1e') ...
            ',obj=' num2str(obj) ',stopALM=' num2str(stopC,'%2.3e')]);
    end
    if stopC<tol 
        break;
    else
        %update lagrage multiplier
        Y = Y + mu*leq;
        %update penalty parameter
        mu = min(max_mu,mu*rho);
    end
end

%% Denormalization
M_est = UV+b*ones(1,n);

end
