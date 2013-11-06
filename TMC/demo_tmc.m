function demo_tmc()
%%  This script is a demo of tmc method for tag completion problem in PAMI2013

%% Initialize configuration: path, parameters
COREL5K_PATH = 'C:\workspace\program\image-annotation\icme2014\tag-completion\completion\svm-vt\mat_corel5k\';


param.lambda = 1;
param.mu = 1;
param.ita = 1;
param.gamma = 10;


%% the tmc algorithm, style as pegasos
% input: T_cap~ nxm, param, thres
maxIter = 100;
Tolerance = 10^-3;

load([COREL5K_PATH, 'corel5k_DenseSift_train.mat']); %nxd
V = DenseSift_train;
load([COREL5K_PATH, 'corel5k_DenseSift_test.mat']); %nxd
Vt = DenseSift_test;

load([COREL5K_PATH, 'corel5k_test_annot.mat']);
yTe_gt = double(test_annot);

load([COREL5K_PATH, 'corel5k_anno_50.mat']); % nxm
T_cap_train = yTrp;
T_cap_test = yTep;

[n_tr, d] = size(V);
[n_te, d] = size(Vt);
[n_tr, m] = size(T_cap_train);
[n_te, m] = size(T_cap_test);

n = n_tr + n_te;

V = [V; Vt];
T_cap = [T_cap_train; zeros(size(T_cap_test))];

test_idx = [n_tr+1 : n];

R = T_cap'*T_cap;






wt = ones(d,1);
Tt = T_cap;

Topt = [];
wopt = [];

Loss = zeros(maxIter, 1);
Loss(1) = calc_loss(Tt, T_cap, V, R, wt, param);

for t = 1 : maxIter
    stepsize = 1 / (t+1);
    G = (Tt * Tt') - V*diag(wt)*V';
    H = (Tt'*Tt - R);
    gT = 2*G* Tt + ...
        2*param.lambda*Tt*H + ...
        2*param.ita*(Tt - T_cap);
    
    gW = -2*diag(V'*G*V);
    
    Tt1_cap = Tt - stepsize * gT;
    wt1_cap = wt - stepsize * gW;
    
    Tt1 = Tt1_cap - param.mu * stepsize * ones(n,m);
    wt1 = wt1_cap - param.gamma * stepsize * ones(d,1);
    
    Tt1(Tt1 < 0) = 0;
    wt1(wt1 < 0) = 0;
    
    Tt = Tt1;
    wt = wt1;
    
    Loss(t+1) = calc_loss(Tt1, T_cap, V, R, wt1, param);
    
    ratio = norm((Loss(t+1) - Loss(t)) / Loss(t));
    
    if  ratio < Tolerance || t == maxIter
        Topt = Tt1;
        wopt = wt1;
        break
    end
    fprintf('Iteration %d, loss %f, ratio %f \n', t, Loss(t+1), ratio);
    pred_score = Tt1(test_idx, :);
    results = evaluatePR(yTe_gt', pred_score',5);
    fprintf('... P %f, R %f, N+ %d \n', results.prec, results.rec, results.retrieved);
end

fprintf('tmc method finished!\n');






end


function res = calc_loss(T, T_cap, V, R, w, param)
l1 = T*T' - V * diag(w) * V';
l2 = T'*T - R;
l3 = T - T_cap;

res = norm(l1, 'fro') + param.lambda * norm(l2, 'fro') + ...
    param.ita * norm(l3, 'fro') + param.mu*norm(T, 1) + param.gamma*norm(w,1);

end