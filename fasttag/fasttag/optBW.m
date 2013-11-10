function [Ms, Ws, Bs] = optBW(Lfreq, L, alpha, noise, maxLayer, weights, invTr, Xb, rWfreq, rW)

lambda = 1e-6;
maxIter = 10;
tol = 1e-2;

d = size(Xb, 1);
[r, n] = size(Lfreq);
M = L;

[B0s] = optB(Lfreq, L, weights,  noise, maxLayer, lambda, rWfreq, rW);
Bs = B0s;
weights = spdiags(weights', 0, length(weights), length(weights));

for ii = 1:maxLayer
	k = size(M, 1);
	iB = eye(k+1);
	iB(end, end) = 0;

	Mb = [M; ones(1, n)]; % k+1 x n
	weightedMb = weights*Mb'; % n x k+1
    
% 	Sl = Mb*weightedMb; % k+1 x k+1
%     Sl = updateSl(Sl, rWfreq, rW);
    
    Sl = updateSl_(Mb, weightedMb, weights, rWfreq, rW);
    
	q = ones(k+1, 1)*(1-noise);
	q(end) = 1;

	Q = Sl.*(q*q');
    Q(1:k+2:end) = q.*diag(Sl);

    if ii == 1
%         P = Lfreq*weightedMb.*repmat(q', r, 1);
        P = updateP(Lfreq*weightedMb, rW, rWfreq);
    else
        P = Sl(1:end-1, :).*repmat(q', k, 1);
    end

	B = B0s{ii};
	prevB = B;
	prevW = rand(r, d);

	Td = Mb*invTr;
	for iter = 1:maxIter
		W = B*Td;
		pred = W*Xb;
		B = (alpha*P + pred*weightedMb)/(alpha*Q + alpha*lambda*iB + Sl);

		optcondW = norm(W-prevW, 'fro')/norm(prevW, 'fro');
                optcondB = norm(B-prevB, 'fro')/norm(prevB, 'fro');
		if optcondW < tol && optcondB < tol
            break;
        end
        prevW = W;
        prevB = B;
    end
	M = tanh(B*Mb);
	Ws{ii} = W;
	Ms{ii} = M;
end

end

function S = updateSl(Sl, rWfreq, rW)
% s1 mxn
S = zeros(size(Sl));
N = size(Sl, 2);
if size(Sl,1) == size(rW,1) + 1
    for n = 1 : N
        M = [rW(:,n);1];
        S = S + diag(M)*Sl*diag(M)';
    end
    
elseif size(Sl,1) == size(rWfreq,1) + 1
    for n = 1 : N
        M = [rWfreq(:,n);1];
        S = S + diag(M)*Sl*diag(M)';
    end
end
S = S ./ N;    
end

function S = updateSl_(M1, M2, weights, rWfreq, rW)
% M1 is k+1 x n , M2 is n x k+1 weights is nxn diag
% rWfreq is k x n, rW is L x n
[K, N] = size(M1);
S = zeros(size(K,K));

if size(M1,1) == size(rW,1) + 1
    for n = 1 : N
        Sl = weights(n,n) * M1(:,n) * M2(n,:); % KxK
        M = [rW(:,n);1];
        S = S + diag(M)*Sl*diag(M)';
    end
    
elseif size(M1,1) == size(rWfreq,1) + 1
    for n = 1 : N
        Sl = weights(n) * M1(:,n) * M2(n,:); % KxK
        M = [rWfreq(:,n);1];
        S = S + diag(M)*Sl*diag(M)';
    end
end

end


function P = updateP(P1, rW, rWfreq)
% P kxk rWfreq kxN
P = zeros(size(P1));
N = size(rWfreq,2);

if size(P1, 2) == size(rWfreq,1) + 1
    for n = 1 : N
        M = [rWfreq(:,n);1];
        P = P + diag(rWfreq(:,n))*P1*diag(M)';
    end
elseif size(P1, 2) == size(rW,1) + 1
    for n = 1 : N
        M = [rWfreq(:,n);1];
        T = [rW(:,n);1];
        P = P + diag(rWfreq(:,n))*P1*diag(T)';
    end
end

P = P ./ N;
end