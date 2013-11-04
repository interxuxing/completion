function [] = demo_svm()
% This function is a demo code for conventional svm
clc;
%% first config mat file path, load needed data
COREL5K_PATH = 'C:\workspace\program\image-annotation\weakly learning\RelatedCode\fasttag\data\corel5k\';

%load original train&test features and labels information
load([COREL5K_PATH, 'data,dimen=1000.mat']);


% transpose the matrixs, single -> double
xTr = [xTr; ones(1, size(xTr, 2))];
xTe = [xTe; ones(1, size(xTe, 2))];

xTr = double(xTr');
xTe = double(xTe');
yTr = double(yTr');
yTe = double(yTe');




partial_rate = [90, 70, 50, 30, 10];

for p = 1 : length(partial_rate)
% load pairtial label information, this are provided as labels instead of
% original labels
temp_name = sprintf('corel5k_anno_%d.mat', partial_rate(p));
load([COREL5K_PATH, temp_name]);

yTr = double(yTrp');

%% now train L classifiers use libsvm
options.kernel = 'linear'; %linear, RBF
options.platt = 0;
    
if 0
    [models] = svm_train(xTr, yTrp, valIdx, options);
    save(fullfile(COREL5K_PATH, 'models.mat'),'models');
else
    load(fullfile(COREL5K_PATH, 'models.mat'));
end

%% now used learnt models for test samples
%% load feature
[test_labels,test_outputs] = svm_test(xTe,yTe,...
    xTr,models, options);


%% evaluation on test with ground-truth, AUC, ROC
yTest = yTe;
yTest(yTest == 0) = -1;
[tpr, fpr] = evalROC(test_outputs, yTest);
[area, area2] = evalAUC(fpr, tpr);

test_labels(test_labels == -1) = 0;
[results]= evaluatePR(yTe', test_outputs', 5);

fprintf('Prec %f, Rec %f, F1 %f, N+ %d \n', prec, rec, f1, retrieved);


end


end