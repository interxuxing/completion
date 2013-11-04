function [] = demo_svm_vt()
% This function is a demo code for svm_vt
clc;
%% first config mat file path, load needed data
COREL5K_PATH = fullfile(eval('pwd'), 'mat_corel5k');

%load train features and labels information
load(fullfile(COREL5K_PATH, 'corel5k_DenseSift_train.mat'));
train_sift = normalize_image(DenseSift_train, [0,1]);
[N, D] = size(train_sift);

load(fullfile(COREL5K_PATH, 'corel5k_train_annot.mat'));
y_train = double(train_annot);

[N, L] = size(y_train);

if 0
% now generate label-specific positive / negative indexes
for l = 1 : L
    label_indexes{l}.pos = (y_train(:, l) == 1);
    label_indexes{l}.neg = (y_train(:, l) ~= 1);
end


% calculate pair-wise distance for all training samples and normalize
% all elemtents to range [0,1]
% for viusal distance, NxN
visual_dist = slmetric_pw(train_sift', train_sift', 'chisq');
% now scale according to row
row_max = max(visual_dist,[],1);
visual_dist_norm = 1 - visual_dist ./ repmat(row_max,N,1);

semantic_dist = slmetric_pw(y_train, y_train, 'nrmcorr');
row_max = max(semantic_dist,[],1);
semantic_dist_norm = semantic_dist ./ repmat(row_max,L,1);

% save needed infor
vt_info_file = fullfile(COREL5K_PATH, 'vt_info.mat');
save(vt_info_file,'label_indexes', 'visual_dist','visual_dist_norm',...
    'semantic_dist','semantic_dist_norm');

else
    load(fullfile(COREL5K_PATH, 'vt_info.mat'));
end
%% generate tolorrence values for each samples in each label
if 0
    tolerance_value = generate_tolorrence(label_indexes, train_sift, y_train, ...
        visual_dist_norm, semantic_dist_norm);
    vt_info_file = fullfile(COREL5K_PATH, 'vt_info.mat');
    save(vt_info_file,'tolerance_value','-append');
%else
    load(fullfile(COREL5K_PATH, 'vt_info.mat'));
end
%% now train L classifiers use libsvm
if 1
    options.kernel = 'linear';
    options.platt = 0;
    
    [models] = svm_vt_train(train_sift, y_train, tolerance_value, options);
    save(fullfile(COREL5K_PATH, 'models.mat'),'models');
else
    load(fullfile(COREL5K_PATH, 'models.mat'));
end

%% now for test set
%% load feature
load(fullfile(COREL5K_PATH, 'corel5k_DenseSift_test.mat'));
test_sift = normalize_image(DenseSift_test,[0,1]);

load(fullfile(COREL5K_PATH, 'corel5k_test_annot.mat'));
y_test = double(test_annot);

[test_labels,test_outputs] = svm_vt_test(test_sift,y_test,...
    train_sift,models, options);


%% evaluation on test with ground-truth
load(fullfile(COREL5K_PATH, 'corel5k_test_annot.mat'));
gt_test = double(test_annot);

test_labels(test_labels == -1) = 0;
[prec, rec, f1, retrieved, f1Ind, precInd, recInd]= evaluatePR(gt_test', test_outputs', 5);

fprintf('Prec %f, Rec %f, F1 %f, N+ %d \n', prec, rec, f1, retrieved);
end