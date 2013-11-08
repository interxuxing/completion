addpath('util/')
addpath('preprocess/')
addpath('baseline/')
addpath('fasttag/')

topK = 5;
dimen = 1000;


datasets = {'corel5k'; 'espgame'; 'iaprtc12'};

% for i = 1:size(datasets, 1)
for i = 1
	dataset = datasets{i};
	dataFolder = ['./data/', dataset, '/'];

	filename=[dataFolder, 'data,dimen=', num2str(dimen), '.mat']; 
    filename_t = [dataFolder, 'tolerance.mat'];
	if exist(filename)
		load(filename);
        load(filename_t);
	else
		% preprocessing, includes approximated additive kernel mapping and random projection to reduce dimension
		[xTr, yTr, xTe, yTe, valIdx] = loaddata(dataFolder, dimen);	
	end

	xTr = double(xTr(1:200,:));
	xTe = double(xTe(1:200,:));
	yTr = double(yTr);
	yTe = double(yTe);

 
    filename = [dataFolder, 'reweights.mat'];
    if exist(filename,'file')
        load(filename);
    else
        reweights = imagetagreweighting(yTr, xTr);
        save(filename, 'reweights');
    end
    
    
	xTr = [xTr; ones(1, size(xTr, 2))];
	xTe = [xTe; ones(1, size(xTe, 2))];

	%linear regression baseline
% 	[W_lr] = linear_regression(xTr, yTr, xTe, yTe, topK);
%     yTr = tolerance_value';

%%  train each W vector for each tag 
%     [D, N] = size(xTr);
%     [T, N] = size(yTr);
%     W_lrt = zeros(T, D);
%     for t = 1 : T
%         xTr_vt = xTr .* repmat(tolerance_value(:,t),1,D)';
%         yTr_vt = yTr(t,:);
%         
%         beta = 1e-2;
% 
%         [d, n] = size(xTr);
%         Wpos = 1./max(1, sum(yTr_vt>0, 2));
%         weights = max(bsxfun(@times, (yTr_vt>0), Wpos), [], 1);
% 
%         iW = spdiags([ones(d-1, 1); 0], 0, d, d); % make a diag matrix on the 0-th diagonals
%         Sx = xTr*spdiags(weights', 0, n, n)*xTr';
%         W_lrt(t,:) = yTr_vt*spdiags(weights', 0, n, n)*xTr'/(Sx+beta*iW);
%         
%     end
%     
%     predTe = W_lrt*xTe;
%     [prec, rec, f1, retrieved] = evaluate(yTe, predTe, topK);
%     fprintf('\nLinearRegression :: Prec = %f, Rec = %f, F1 = %f, N+ = %d\n',  prec, rec, f1, retrieved);
    
	% fasttag
    
    
	[W_fasttag] = fasttag(xTr, yTr, xTe, yTe, topK, valIdx, reweights);

end
