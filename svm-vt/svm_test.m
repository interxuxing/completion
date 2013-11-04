function [test_labels,test_outputs] = svm_test(test_data,gt_labels,...
    train_data,models,options)
% This function is to classfiy test samples with learnt classifiers

[N_test, D] = size(test_data);
[N_test, L] = size(gt_labels);


%% set pre-parameters
kernel_type = options.kernel;
prob_platt = options.platt;
para_sigma = 1;

%% calculate kernel matrix
if(strcmp(kernel_type,'linear'))
    kernelX=test_data * train_data';
elseif(strcmp(kernel_type,'RBF'))
    kernelX=RBF_kernel(test_data,para_sigma,train_data);
else
    kernelX = test_data;
end
kernel_sum=[(1:N_test)',kernelX];

%% predict
test_labels=zeros(N_test,L);
test_outputs=zeros(N_test,L);
for i = 1 : L
    test_targets = ones(N_test,1);
    test_targets(gt_labels(:,i) == 0) = -1;
    
    [test_label,accuracy,test_output] = svmpredict(test_targets,kernel_sum,models{i}.model,'-b 1');
%     if(test_label(1)*test_output(1)<0)
%         test_output=-test_output;
%     end
    
    if ~isempty(test_label)
        test_labels(:,i)=test_label;
        test_outputs(:, i) = test_output(:, models{i}.model.Label == 1);
    else
        test_labels(:,i) = -1 * ones(N_test,1);
        test_outputs(:, i) = zeros(size(test_targets));
    end

%     if prob_platt
%         test_outputs(:,i) = fxprob(test_output, models{i}.A, models{i}.B);
%     else
% %         test_outputs(:,i) = test_output(:,1);
%         test_outputs(:, i) = test_output(:, models{i}.model.Label == 1); 
%     end
end

end

function [prob] = fxprob(vect, A, B)
    prob = 1 ./ (1 + exp(A .* vect + B));
end