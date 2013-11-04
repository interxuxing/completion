function [models] = svm_train(train_feature, train_tags, valIdx, options)
% This function is to train conventional svm classifer
% train_data, train_labels are NxD matrix

train_data = train_feature(~valIdx, :);
train_labels = train_tags(~valIdx, :);
val_data = train_feature(valIdx, :);
val_labels = train_tags(valIdx, :); 

[N_Tr, D] = size(train_data);
[N_Tr, L] = size(train_labels);
[N_Val, D] = size(val_data);
[N_Val, L] = size(val_labels);

%% first set some default parameters
lambda = 10;
kernel_type = options.kernel;
prob_platt = options.platt;
is_cv = options.cv; 
para_sigma = 1;

%% then calculate kernel matrix
%   now compute kernel matrix
if(strcmp(kernel_type,'linear'))
    kernelX = train_data * train_data';
    kernelXX = val_data * train_data';
elseif(strcmp(kernel_type,'RBF'))
    kernelX=RBF_kernel(train_data, para_sigma);
    kernelXX = RBF_kernel(val_data,para_sigma,train_data);
else
    kernelX = train_data;
    kernelXX = val_data;
end
kernel_sum = [(1:N_Tr)', kernelX + 0.00001*eye(size(kernelX))]; % avoid low rank
kernel_sumXX = [(1:N_Val)', kernelXX + 0.00001*eye(size(kernelXX))]; % avoid low rank

models = cell(1,L);


% now train classifier for each label
%% first we need modify the original feature
for l = 1 : L
    train_targets = ones(size(train_labels,1),1);
    train_targets(train_labels(:,l) == 0) = -1;
     
    val_targets = ones(size(val_labels,1),1);
    val_targets(val_labels(:,l) == 0) = -1;
    % now use libsvm 
    fprintf('...train svm classifier for %d-th label ... \n', l);
    prior_pos = length(find(train_targets == 1)); 
    prior_neg = length(find(train_targets ~= 1));
    fprintf('... ... Num of pos: %d, neg: %d \n', prior_pos, prior_neg);
    ratio = floor(prior_neg / (prior_pos+1));
    
    model = svmtrain(train_targets,kernel_sum,['-t 4 -q -b 1 -c ',num2str(lambda),...
    ' -w1 ', num2str(ratio), ' -w-1 1']);
        
    if is_cv
        param.ratio = ratio;
        param.c_list = [-5:1:5];
        param.g_list = [-5:1:5];
        optimals = cv_train(train_targets, kernel_sum, param);
        fprintf('... ... %d-th classifier, optimal C %f, optimal g %f \n', l, optimals.c, optimals.g);
        model = svmtrain(train_targets, kernel_sum, ...
            ['-q -t 4 -b 1 -c ', num2str(optimals.c), ' -g ', num2str(optimals.g)]);
%         [tmp1,tmp2,dec_val] = svmpredict(val_targets, kernel_sum, model, '-b 1');
    else
        model = svmtrain(train_targets, kernel_sum, ['-t 4 -q -b 1 -c 10 -g 0.2']);
    end
    
    [~,~,p] = svmpredict(val_targets, kernel_sumXX, model, '-b 1');
%     prob = p(:,model.Label==1);

    if prob_platt
        % now use platt to calibrate the dec_value to probabilities
%         prior_pos = length(find(train_targets == 1)) / length(train_targets); 
%         prior_neg = length(find(train_targets ~= 1)) / length(train_targets); 
        [A, B] = platt(dec_val, train_targets, prior_neg, prior_pos);
        models{l}.A = A;
        models{l}.B = B;
    else
        models{l}.model = model;
    end
end


end


function [optimals] = cv_train(train_label, train_data, options)
% do cross validation to find best C and gamma
bestcv = 0; 
bestc = 0;
bestg = 0;

numLog2c = length(options.c_list);
numLog2g = length(options.g_list);
cvMatrix = zeros(numLog2c, numLog2g);

for i = 1: numLog2c
    for j = 1: numLog2g
        log2g = options.g_list(j);
        log2c = options.c_list(i);
        param = ['-q -t 4 -v 3 -b 1 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)]; ...
%             ' -w1 ', num2str(options.ratio), ' -w-1 1'];
        cv = svmtrain(train_label, train_data, param);
        cvMatrix(i, j) = cv;
        if cv >= bestcv
            bestcv = cv;
            bestc = 2^log2c;
            bestg = 2^log2g;
            bestparam = param;
        end
    end
end

optimals.c = bestc;
optimals.g = bestg;

fprintf('...... cv finished! \n C %f, gamma %f \n best param %s \n',...
    optimals.c, optimals.g, bestparam);

end