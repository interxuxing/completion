function to_val = generate_tolorrence(index, feature_matrix, label_matrix, ...
    visual_dist, semantic_dist)
% This function is to generate the tolorrence values for each positive
% negative sample in each label set.
%
% detailed tolorrence value is calculated from visual similarity and 
% label correlations.
%
% eg.   index is a cell, sizeof L
%       [N, D] = size(feature_matrix); [N, L] = size(label_matrix)
%       [N, N] = size(visual_dist); [L, L] = size(semantic_dist);
%   then to_val(pos) = 1;
%        to_val(neg) = 1 - 1/2 * score(visual) - 1/2 * score(semantic)
%
% return to_val is NxL matrix of tolorrence values.

[N, D] = size(feature_matrix); [N, L] = size(label_matrix);

to_val = ones(N, L);


for l = 1 : L
    pos_index = find(index{l}.pos == 1);
    neg_index = find(index{l}.neg == 1);
    vt = [];
    for n = 1 : length(neg_index)
        % visual similarity
        vt_vis = max(visual_dist(neg_index(n), index{l}.pos),[],2);
        
        % semantic similarity
        L_n = (label_matrix(neg_index(n),:) ~= 0);     
        vt_sem = max(semantic_dist(L_n, l));
        if isempty(vt_sem)
            vt_sem = 0;
        end
        try
            vt(n) = 1 - (vt_vis + vt_sem) / 2;
        catch
            lasterr;
        end
    end
    
    try
    to_val(neg_index, l) = vt;
    catch
        lasterr;
    end
end
