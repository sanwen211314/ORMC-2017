%% Test MC on generated data

function Out = TestMConGeneratedData()
% mxn data matrix M with rank r
m = 500;
n = 500;
r = 10;

U0 = rand(m,r);
V0 = rand(r,n);

W = double(rand(m,n) > 0.5);  % observed matrix
Omega = find(W);

%ground truth 
M0 = U0*V0; %+ones(m,n);
% adding column outiers
M = M0;
ratio = 0.5;
CoutN = round(n*ratio);
Omat = randn(m,CoutN)*2;
M(:,end - CoutN + 1:end) = M(:,end - CoutN + 1:end) + Omat;

%% Test

tic
out = ORMC(M.*W,W,r,200);
% plot objective value
a = out.re;
plot(1:length(a),a);
%plot weighted of all columns
weighted = out.weight;
plotweighted(weighted)
%test re on inliers
M_est = out.matrix;
toc
E = M0 - M_est;
E(:,end - CoutN + 1:end) = [];
RE = norm(E,'fro')/norm(M0(:,1:end - CoutN),'fro');
Out.ORMC = RE;
Out.weighted = weighted;
end

















