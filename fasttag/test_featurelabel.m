addpath('util/')
addpath('preprocess/')
addpath('baseline/')
addpath('fasttag/')

topK = 5;
dimen = 1000;


datasets = {'corel5k'; 'espgame'; 'iaprtc12'};

for i = 1:size(datasets, 1)

	dataset = datasets{i};
	dataFolder = ['./data/', dataset, '/'];

	filename=[dataFolder, 'data,dimen=', num2str(dimen), '.mat']; 
    
	if exist(filename)
		load(filename);
        
	else
		% preprocessing, includes approximated additive kernel mapping and random projection to reduce dimension
		[xTr, yTr, xTe, yTe, valIdx] = loaddata(dataFolder, dimen);	
	end

	xTr = double(xTr);
	xTe = double(xTe);
	yTr = double(yTr);
	yTe = double(yTe);

	xTr = [xTr; yTr; ones(1, size(xTr, 2))];
	xTe = [xTe; yTe; ones(1, size(xTe, 2))];

    
    
	%linear regression baseline
	[W_lr] = linear_regression(xTr, yTr, xTe, yTe, topK);


    
	% fasttag
% 	[W_fasttag] = fasttag(xTr, yTr, xTe, yTe, topK, valIdx);

end
