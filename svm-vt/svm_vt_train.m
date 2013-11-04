function [models] = svm_vt_train(train_data, train_labels, tolerance, options)
% This function is to train svm-vt classifiers for each label
%   tolerance decides each negative sample's weight in hinge loss

[N, D] = size(train_data);
[N, L] = size(train_labels);
[N, L] = size(tolerance);

%% first set some default parameters
lambda = 1;
kernel_type = options.kernel;
prob_platt = options.platt;

models = cell(1,L);

% now train classifier for each label
%% first we need modify the original feature
for l = 1 : L
    vt_l = tolerance(:,l);
    % multiple the original feature with vt
    train_feature = train_data .* repmat(vt_l,1,D);
    train_targets = ones(size(train_labels,1),1);
    train_targets(train_labels(:,l) == 0) = -1;
    
    %   now compute kernel matrix
    if(strcmp(kernel_type,'linear'))
        kernelX = train_feature * train_feature';
    elseif(strcmp(kernel_type,'RBF'))
        kernelX=RBF_kernel(train_feature,sigma);
    else
        kernelX = train_feature;
    end
    kernel_sum = [(1:N)', kernelX + 0.00001*eye(size(kernelX))]; % avoid low rank
    
    % now use libsvm 
    fprintf('...train %d-th label ... \n');
    model = svmtrain(train_targets,kernel_sum,['-t 4 -c ',num2str(lambda)]);
    [tmp1,tmp2,dec_val] = svmpredict(train_targets,kernel_sum,model);
    
    % now use platt to calibrate the dec_value to probabilities
    prior_pos = length(find(train_targets == 1)) / length(train_targets); 
    prior_neg = length(find(train_targets ~= 1)) / length(train_targets); 
    
    if prob_platt
        [A, B] = platt(dec_val, train_targets, prior_neg, prior_pos);
        models{l}.A = A;
        models{l}.B = B;
    else
        models{l}.model = model;
    end
end



end