function [Pf,Pt,Pz] = PLCA(V,R,Niter,parameter)
% [Pf,Pt,Pz] = PLCA(V,R,Niter,parameter)
% PLCA as proposed by Smaragdis :
%   input : - V : non-negative matrix to be decomposed.
%           - R : number of atoms
%           - Niter : number of iterations
%           - parameter (optional) : additional parmeters. A structure with fields Pf
%           (initialization of Pf), Pt (initialization of Pt), Pz
%           (initialization of Pz), alphaf (Dirichlet prior on Pf), muf
%           (weight of the prior on Pf), alphat (Dirichlet prior on Pt),
%           mut (weight of the prior on Pt), alphaz (Dirichlet prior on Pz),
%           muz (weight of the prior on Pz). Every field is optional
%   output : - Pf : frequency templates
%            - Pt : activations
%            - Pz : weight of each component
%
% Copyright (C) 2010 Romain Hennequin


plotResults = true;
M = size(V,1);
N = size(V,2);

if nargin == 3
    parameter = struct();
end

% initialization of parameters
if isfield(parameter,'Pf')
    Pf = parameter.Pf;
else
    Pf = rand(M,R);
    Pf = Pf./(ones(M,1)*sum(Pf));
end

if isfield(parameter,'Pt')
    Pt = parameter.Pt;
else
    Pt = rand(R,N);
    Pt = Pt./(sum(Pt,2)*ones(1,N));
end

if isfield(parameter,'Pz')
    Pz = parameter.Pz;
else
    Pz = ones(R,1);%rand(R,1);
    Pz = Pz./sum(Pz);
end
diagPz = diag(Pz);

if isfield(parameter,'alphaf')
    alphaf = parameter.alphaf;
    if isfield(parameter,'muf')
        muf = parameter.muf;
    else
        muf = 1;
    end
else
    alphaf = zeros(M,R);
    muf = 0;    
end

if isfield(parameter,'alphat')
    alphat = parameter.alphat;
    if isfield(parameter,'mut')
        mut = parameter.mut;
    else
        mut = 1;
    end
else
    alphat = zeros(R,N);
    mut = 0;    
end

if isfield(parameter,'alphaz')
    alphaz = parameter.alphaz;
    if isfield(parameter,'muz')
        muz = parameter.muz;
    else
        muz = 1;
    end
else
    alphaz = zeros(R,1);
    muz = 0;    
end

% Waitbar
message = ['computing PLCA. iteration : 0/' int2str(Niter) ];
h = waitbar(0,message);
if plotResults
    fig = figure;
end

oneNN = ones(N);
oneMM = ones(M);
LL = -inf*ones(Niter,1);
sumV = sum(V(:));


muf0 = muf;
mut0 = mut;
muz0 = muz;

% iterations
for n = 1:Niter

    % decreasing prior
    muf = muf0*(Niter-n)/Niter;
    mut = mut0*(Niter-n*0.9)/Niter;
    muz = muz0*(Niter-n)/Niter;
    
    % decreasing prior
    Pftemp = Pf;
    Pttemp = Pt;
    Pztemp = Pz;
    
    % spectral templates

    Lambdatemp = Pftemp*diag(Pztemp)*Pttemp;
    VonLambda = V./(Lambdatemp+eps);

    Pf = Pf.*(VonLambda*Pttemp'*diag(Pztemp)) + alphaf*diag(muf);
    Pf = Pf ./ (oneMM*Pf + eps);

%     Pftemp = Pf;
%     Pttemp = Pt;
%     Pztemp = Pz;
    
    if sum(isnan(Pf(:)))
        disp('Pf is nan')
    end
    
    % activation
    Lambdatemp = Pftemp*diag(Pztemp)*Pttemp;
    VonLambda = V./(Lambdatemp+eps);
    
    Pt = (Pt).*(diag(Pztemp)*Pftemp'*VonLambda ) + diag(mut)*alphat;
    Pt = Pt ./ (Pt*oneNN + eps);
    
    mx = max(Pt(:));
    Pt(Pt<mx*0.0001) = 0;
%     Pftemp = Pf;
%     Pttemp = Pt;
%     Pztemp = Pz;
    
    Lambdatemp = Pftemp*diag(Pztemp)*Pttemp;
    
 
    % relative power of templates
    Pz = Pz.* diag(Pftemp'*VonLambda*Pttemp') + muz*alphaz;
    Pz = Pz./(sum(Pz));
    
    if sum(isnan(Pz(:)))
        disp('Pz is nan')
    end

    Lambda = Pf*diag(Pz)*Pt;

    message = ['computing PLCA. iteration : ' int2str(n) '/' int2str(Niter)];    
    LL(n) = sum(V(:).*log(Lambda(:)+eps));  

    
    % plot
    if mod(n,10)==1
        if plotResults
            figure(fig)
            subplot(221)
            imagesc(db(Lambda))
            caxis([db(max(Lambda(:)))-100,db(max(Lambda(:)))])
            axis xy
            colorbar
            title('Reconstructed spectrogram')
            subplot(222)
            plot(db(Pz),'o-')
            title('Relative weight of each component')
            subplot(223)
            plot(LL,'o-')
            title('Log-likelihood')
            subplot(224)
            imagesc(db(Pt))
            caxis([db(max(Pt(:)))-100,db(max(Pt(:)))])
            colorbar
            title('Activations')
        end
        waitbar(n/Niter,h,message);
    end
    disp(message);
end

close(h)
if plotResults
    close(fig)
end