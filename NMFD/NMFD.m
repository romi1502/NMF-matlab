function [W, H] = NMFD(V,R,T,Niter,initVal)
% [W, H] = NMFD(V,R,T,Niter,initVal)
%    NMFD as proposed by Smaragdis (Non-negative Matrix Factor
%    Deconvolution; Extraction of Multiple Sound Sources from Monophonic
%    Inputs). KL divergence minimization. The proposed algorithm was
%    corrected.
%Input :   
%   - V : magnitude spectrogram to factorize (is a MxN matrix)
%   - R : number of templates
%   - T : template size (in number of frames in the spectrogram)
%   - Niter : number of iterations
%   - initVal (optional) : initialization for W and H. A structure with
%   fields W (initial value of W) and H (initial value of H).
%Output :
%   - W : time/frequency template (TxMxR array, each template is TxM)
%   - H : activities for each template (RxN matrix)
%
% Copyright (C) 2010 Romain Hennequin



% V : spectrogram MxN
% H : activation RxN
% Wt : spectral template MxR t = 0:T-1
% W : TxMxR


% data size
M = size(V,1);
N = size(V,2);

% initialization
if nargin<5
    H = rand(R,N);
    W = rand(M,R,T);
    One = ones(M,N);
    Lambda = zeros(M,N);
else
    H = initVal.H;
    W = initVal.W;
    One = ones(M,N);
    Lambda = zeros(M,N);
end

% Waitbar
message = ['computing NMFD. iteration : 0/' int2str(Niter)];
h = waitbar(0,message);

for iter =1:Niter
    
    % computation of Lambda
    Lambda(:) = 0;
    for f = 1:M
        for z = 1:R
            v = reshape(W(f,z,:),T,1);
            cv = conv(v,H(z,:));
            Lambda(f,:) = Lambda(f,:) + cv(1:size(Lambda,2));
        end
    end

    
    Halt = H;
    
    Htu = zeros(T,R,N);
    Htd = zeros(T,R,N);
   
    % update of H for each value of t (which will be averaged)
    VonLambda = V./(Lambda + eps);
    
    Hu = zeros(R,N);    
    Hd = zeros(R,N);    
    for z = 1:R
        for f = 1:M
            v = reshape(W(f,z,:),T,1);
            cv = conv(VonLambda(f,:),flipud(v));
            Hu(z,:) = Hu(z,:) + cv(T:T+N-1);
            
            v = reshape(W(f,z,:),T,1);
            cv = conv(One(f,:),flipud(v));
            Hd(z,:) = Hd(z,:) + cv(T:T+N-1);
        end
    end
    
    
    % average along t
    H = H.*Hu./Hd;

    figure(20)
    subplot(211)
    imagesc(db(V));
    axis xy
    ax = caxis;
    title('Original spectrogram')
    xlabel('time')
    ylabel('frequency')
    subplot(212)
    imagesc(db(Lambda));
    axis xy
    caxis(ax);
    title('Reconstructed spectrogram')
    xlabel('time')
    ylabel('frequency')
    
% computation of Lambda

    Lambda(:) = 0;
    for f = 1:M
        for z = 1:R
            v = reshape(W(f,z,:),T,1);
            cv = conv(v,H(z,:));
            Lambda(f,:) = Lambda(f,:) + cv(1:size(Lambda,2));
        end
    end


    mu = 1;
    lambda = 1.02.^(0:T-1)-1;
    lambda(1:3) = 0;
    
    SumTot = sum(W,2);
    
    VonLambda = V./(Lambda + eps);
    
    
    % update of Wt
    for t=0:T-1
       W(:,:,t+1) = W(:,:,t+1).*(  (VonLambda*shiftLR(H,-t)') ./ (One*shiftLR(H,-t)'+ eps + mu*lambda(t+1))  );
    end
    
    message = ['computing NMFD. iteration : ' int2str(iter) '/' int2str(Niter)];
    waitbar(iter/Niter,h,message);
end
close(h)