function weights = imagetagreweighting(tagMatrix, featureMatrix)
% function weights = imagetagreweighting(tagMatrix, featureMatrix)
%   is to generate the reweighting tag scores for each image
%   considering visual and semantic cues
%
% INPUT tagMatrix MxN featureMatrix DxN
% OUTPUT weights MxN

[M, N] = size(tagMatrix);
[D, N] = size(featureMatrix);

weights = ones(D, N);

% now generate label-specific positive / negative indexes
for m = 1 : M
    label_indexes{m}.pos = (tagMatrix(m, :) == 1);
    label_indexes{m}.neg = (tagMatrix(m, :) ~= 1);
end

% calculate pair-wise distance for all training samples and normalize
visual_dist = slmetric_pw(featureMatrix, featureMatrix, 'sqdist');
% now scale according to row
row_max = max(visual_dist,[],1);
visual_sim = 1 - visual_dist ./ repmat(row_max,N,1);

semantic_dist = slmetric_pw(tagMatrix', tagMatrix', 'nrmcorr');
row_max = max(semantic_dist,[],1);
semantic_sim = semantic_dist ./ repmat(row_max,M,1);

% reweighting for each tag in each image
for l = 1 : M
    neg_index = find(label_indexes{l}.neg == 1);
    vt = [];
    for n = 1 : length(neg_index)
        % visual similarity
        vt_vis = max(visual_sim(neg_index(n), label_indexes{l}.pos),[],2);
        
        % semantic similarity
        L_n = (tagMatrix(:, neg_index(n))~= 0);     
        vt_sem = max(semantic_sim(L_n, l));
        if isempty(vt_sem)
            vt_sem = 0;
        end
        try
            vt(n) = exp(-(vt_vis + vt_sem)/2) / 2;
        catch
            lasterr;
        end
    end
    
    try
    weights(l, neg_index) = vt;
    catch
        lasterr;
    end
end


end