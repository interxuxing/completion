function [xTr, yTr, xTe, yTe, valIdx] = loaddata(dataFolder, dimen)

[X] = rpSVD(dataFolder, dimen);

matchStr = regexp(ls(dataFolder), '\w*_train_annot.hvecs', 'match');
yTr = vec_read(strcat(dataFolder, matchStr{1}));
yTr = double(yTr');

matchStr = regexp(ls(dataFolder), '\w*_test_annot.hvecs', 'match');
yTe = vec_read(strcat(dataFolder, matchStr{1}));
yTe = double(yTe');

nTr = size(yTr, 2);
nTe = size(yTe, 2);

xTr = X(:, 1:nTr);
xTe = X(:, nTr+1:end);
clear('X');

valIdx = false(1, nTr);
valIdx(randsample(nTr, nTe)) = true;

save([dataFolder, 'data,dimen=', num2str(dimen), '.mat'], 'xTr', 'yTr', 'xTe', 'yTe', 'valIdx', '-v7.3');
