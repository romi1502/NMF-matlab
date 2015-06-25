function [W, H] = NMF2D(V,R,T,Phi,Niter,plotIter)
% [W, H] = NMF2D(V,R,T,Phi,Niter,plotIter)
%    NMF2D as proposed by Morup and Schmidt (Non-negative Matrix Factor 2D
%    Deconvolutionfor Blind Single channel source separation). KL
%    divergence minimization.
% Input :
%   - V : log-frequency magnitude spectrogram to factorize (a MxN matrix)
%   - R : number of templates
%   - T : template time size (in number of frames in the log-frequency spectrogram)
%   - Phi : template frequency size (in number of bin in the log-frequency spectrogram)
%   - Niter : number of iterations
%   - plotIter : plot results for each iteration (value : true or false, default : false)
% Output :
%   - W : time/frequency template (TxMxR array, each template is TxM)
%   - H : time frequency activation for each template (PhixRxN matrix)
%
% Copyright (C) 2010 Romain Hennequin



if nargin == 5
    plotIter = false;
end

% data size
M = size(V,1);
N = size(V,2);

% sparsity of H
lambdaH = 0;
epsilon = 10^-20;

% initialization
H = rand(Phi,R,N);
W = rand(T,M,R);
One = ones(M,N);
Ht = zeros(T,R,N);
Lambda = zeros(M,N);


% Waitbar
message = ['computing NMF2D. iteration : 0/' int2str(Niter)];
h = waitbar(0,message);

for iter =1:Niter

    % computation of Lambda (estimate of V)
    Lambda = 0;
    for t=0:T-1
        for phi = 0:Phi-1
            Lambda = Lambda + shiftUD(reshape(W(t+1,:,:),M,R),phi)* shiftLR(reshape(H(phi+1,:,:),R,N),-t);
        end
    end
    Lambda = Lambda + epsilon;


    % update of W for each t
    denom = zeros(M,R);
    num = zeros(M,R);

    for t=0:T-1
        num = 0;
        denom = 0;
        for phi = 0:Phi-1
            num = num + shiftUD((V./Lambda),-phi) * (shiftLR(reshape(H(phi+1,:,:),R,N),-t)');
            denom = denom + One * (shiftLR(reshape(H(phi+1,:,:),R,N),-t)');
        end
        W(t+1,:,:) = reshape(W(t+1,:,:),M,R).*  (num./denom);
    end

    % recomputation of Lambda (estimate of V)
    Lambda = 0;
    for t=0:T-1
        for phi = 0:Phi-1
            Lambda = Lambda + shiftUD(reshape(W(t+1,:,:),M,R),phi)* shiftLR(reshape(H(phi+1,:,:),R,N),-t);
        end
    end
    Lambda = Lambda + epsilon;



    % update of H for each value of phi
    denom = zeros(R,N);
    num = zeros(R,N);

    for phi=0:Phi-1
        num = 0;
        denom = 0;
        for t = 0:T-1
            num = num + shiftUD(reshape(W(t+1,:,:),M,R),phi)' *shiftLR((V./Lambda),t);
            denom = denom + shiftUD(reshape(W(t+1,:,:),M,R),phi)'*One;
        end
        H(phi+1,:,:) = reshape(H(phi+1,:,:),R,N).* (num./(denom + lambdaH));
    end

    for r=1:R
        normW = sqrt(sum(sum(W(:,:,r).^2)));
        W(:,:,r) = W(:,:,r)/normW;
        H(:,r,:) = H(:,r,:)*normW;
    end


    %%%%% PLOT %%%%%
    
    if plotIter
        
        % templates and activation
        figure(1)
        for r = 1:R
            subplot(2,R,r)
            imagesc(db(reshape(W(:,:,r),T,M))');
            title(['template ' int2str(r)])
            axis xy
            caxis([-100,0])

            subplot(2,R,r+R)
            imagesc(db(reshape(H(:,r,:),Phi,N)));
            title(['activation ' int2str(r)])
            axis xy
            caxis([max(db(V(:)))-50,max(db(V(:)))])

        end

        % (log-freq) spectrogram (original and reconstructed)
        figure(2)
        subplot(211)
        imagesc(db(V))
        title('original magnitude log-frequency spectrogram')
        axis xy
        subplot(212)
        Lambda = zeros(M,N);
        for t=0:T-1
            for phi = 0:Phi-1
                Lambda = Lambda + shiftUD(reshape(W(t+1,:,:),M,R),phi)* shiftLR(reshape(H(phi+1,:,:),R,N),-t);
            end
        end
        imagesc(db(Lambda))
        title('reconstructed log-frequency spectrogram')
        axis xy


    end
    %%%%%%%%%%%%%%



    message = ['computing NMF2D. iteration : ' int2str(iter) '/' int2str(Niter)];
    waitbar(iter/Niter,h,message);
end
close(h)