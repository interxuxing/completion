function [Bs] = optB(Lfreq, L, weights, noise, maxLayers, lambda, rWfreq, rW)

[r, n] = size(Lfreq);
M = L;

weights = spdiags(weights', 0, length(weights), length(weights));
for ii = 1:maxLayers

	k = size(M, 1);
	iB = eye(k+1);
	iB(end, end) = 0;
	Mb = [M; ones(1, n)];

	weightedMb = weights*Mb';
	Sl = Mb*weightedMb;	
    Sl = updateS1(Sl, rWfreq, rW);
	q = ones(k+1, 1)*(1-noise);
	q(end) = 1;

	Q = Sl.*(q*q');
	Q(1:k+2:end) = q.*diag(Sl);

	if ii == 1
% 		P = Lfreq*weightedMb.*repmat(q', r, 1);
        P = updateP(Lfreq*weightedMb, rWfreq);
	else
		P = Sl(1:end-1, :).*repmat(q', k, 1);
	end
	B = P/(Q+lambda*iB);

    Bs{ii} = B;
	M = tanh(B*Mb);

end

end

function S = updateS1(S1, rWfreq, rW)
% s1 mxn
S = zeros(size(S1));
N = size(S1, 2);
if size(S1,1) == size(rW,1) + 1
    for n = 1 : N
        M = [rW(:,n);1];
        S = S + diag(M)*S1*diag(M)';
    end
    S = S / N;
    
elseif size(S1,1) == size(rWfreq,1) + 1
    for n = 1 : N
        M = [rWfreq(:,n);1];
        S = S + diag(M)*S1*diag(M)';
    end
    S = S / N;
end

end

function P = updateP(P1, rWfreq)
% P kxk rWfreq kxN
P = zeros(size(P1));
N = size(rWfreq,2);

for n = 1 : N
    M = [rWfreq(:,n);1];
    P = P + diag(rWfreq(:,n))*P1*diag(M)';
end

P = P / N;
end