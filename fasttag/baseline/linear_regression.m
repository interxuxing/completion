function [W, prec, rec, f1, retrieved] = linear_regression(xTr, yTr, xTe, yTe, topK)
% xTr,xTe DxN, yTr,yTe TxN
beta = 1e-2;

[d, n] = size(xTr);
Wpos = 1./max(1, sum(yTr>0, 2));
weights = max(bsxfun(@times, (yTr>0), Wpos), [], 1);

tic
iW = spdiags([ones(d-1, 1); 0], 0, d, d); % make a diag matrix on the 0-th diagonals
Sx = xTr*spdiags(weights', 0, n, n)*xTr';
W = yTr*spdiags(weights', 0, n, n)*xTr'/(Sx+beta*iW);

% Sx = xTr*xTr';
% W = yTr*xTr'/(Sx+beta*iW);

predTe = W*xTe;
[prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
toc
