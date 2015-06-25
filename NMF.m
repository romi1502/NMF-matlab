function [W, H, bDsave] = NMF(V,R,Niter,beta,initialV)
% [W,H, bDsave] = NMF(V,R,Niter,beta,initialV)
%    NMF with beta divergence cost function.
%Input :
%   - V : power spectrogram to factorize (a MxN matrix)
%   - R : number of templates
%   - Niter : number of iterations
%   - beta (optional): beta used for beta-divergence (default : beta = 0, IS divergence)
%   - initialV (optional) : initial values of W, H (a struct with
%   fields W and H)
%Output :
%   - W : frequency templates (MxR array)
%   - H : temporal activation
%   - bDsave : evolution of beta divergence
%
% Copyright (C) 2010 Romain Hennequin


verbose = false;

eta = 1;

% size of input spectrogram
M = size(V,1);
N = size(V,2);

% initialization
if nargin == 5
    if isfield(initialV,'H')
        H = initialV.H;
    else
        H = rand(R,N);
    end
    if isfield(initialV,'W')
        W = initialV.W;
    else
        W = rand(M,R);
    end
    
    if isfield(initialV,'HRfixed')
        HRfixed = initialV.HRfixed;
    else
        HRfixed = 0;
    end

    if isfield(initialV,'WRfixed')
        WRfixed = initialV.WRfixed;
    else
        WRfixed = 0;
    end

    
else
    H = rand(R,N);
    W = rand(M,R);
    HRfixed = 0;
    WRfixed = 0;
    
    if nargin == 3
        beta = 0;
    end
end

% array to save the value of the beta-divergence
bDsave = zeros(Niter,1);

% computation of Lambda (estimate of V) and of filters repsonse
Lambda = W*H;

% Waitbar
message = ['computing NMF. iteration : 0/' int2str(Niter) ' completed'];
h = waitbar(0,message);


% iterative computation
for iter =1:Niter


%     % plot actual and reconstructed spectrogram
%     figure(22)
%     subplot(211)
%     imagesc(db(V))
%     axis xy
%     title('actual')
%     subplot(212)
%     imagesc(db(Lambda))
%     axis xy
%     title('reconstructed')
%     drawnow;

%     % compute beta divergence and plot its evolution
      bDsave(iter) = betaDiv(V+eps,Lambda+eps,beta);
    figure(23)
    semilogy(bDsave,'-o')
    title(['Evolution of beta divergence (dB) beta = ' num2str(beta) ' eta = ' num2str(eta)])
    xlabel('iteration')
    drawnow;


    % update of W
    if not(WRfixed)
        W = W.* ((Lambda.^(beta-2).*V)*H' +eps)./((Lambda.^(beta-1))*H' + eps);
    else
        W(:,WRfixed+1:end) = W(:,WRfixed+1:end).* ((Lambda.^(beta-2).*V)*H(WRfixed+1:end,:)' +eps)./((Lambda.^(beta-1))*H(WRfixed+1:end,:)' + eps);   
    end
    
    % recomputation of Lambda (estimate of V)
    Lambda = W*H + eps;
    
    
    % update of H
    if not(HRfixed)
        H = H.* (W'*(Lambda.^(beta-2).*V) +eps)./(W'*(Lambda.^(beta-1)) + eps);
    else
        H(1:HRfixed,:) = H(1:HRfixed,:).* (W(:,1:HRfixed)'*(Lambda.^(beta-2).*V) +eps)./(W(:,1:HRfixed)'*(Lambda.^(beta-1)) + eps);
    end
    % recomputation of Lambda (estimate of V)
    Lambda = W*H + eps;


    message = ['computing NMF. iteration : ' int2str(iter) '/' int2str(Niter)];
    if verbose
        disp(message);
    end
    waitbar(iter/Niter,h,message);
end

% % normalization
% for r0=1:R
%     % normalization of templates
%     chosenNorm = 2;
%     normW = norm(W(:,r0),chosenNorm);
%     H(r0,:) = normW*H(r0,:);
%     W(:,r0) = W(:,r0)/normW;
% end

close(h)
close


function bD = betaDiv(V,Vh,beta)
if beta == 0
    bD = sum((V(:)./Vh(:))-log(V(:)./Vh(:)) - 1);
elseif beta == 1
    bD = sum(V(:).*(log(V(:))-log(Vh(:))) + Vh(:) - V(:));
else
    bD = sum(max(1/(beta*(beta-1))*(V(:).^beta + (beta-1)*Vh(:).^beta - beta*V(:).*Vh(:).^(beta-1)),0));
end

