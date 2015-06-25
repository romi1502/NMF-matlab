% test script for NMFD : decomposition of simple drum loops
%
% Copyright (C) 2010 Romain Hennequin

clear all
close all

%% parameters

% Name of the sound file to be decomposed
fileName = 'DL6.wav';

% length of the FFT
Nfft = 1024;

% number of time-frequency atoms
R = 3;

% time-length of the atoms
T = 120;

% number of iterations
Niter = 20;

%% preparing the data

% Load the file
[s,sr] = wavread(fileName);
s = toMono(s);

% compute the spectrogram
sp = stft(s,Nfft,hamming(Nfft,'periodic'),Nfft/4);
V = abs(sp);
M = size(V,1);
N = size(V,2);


%% first decomposition
[W,H] = NMFD(V,R,T,Niter);

%% post processing
initVal.W = W;
initVal.H = max(H,max(H(:))/10)-max(H(:))/10+0.00001;

%% second decomposition
[W,H] = NMFD(V,R,T,10,initVal);


%% separation of the sounds of each component
Lambda = cell(R,1);
for z = 1:R
    Lambda{z} = zeros(M,N);
    for f = 1:M
        v = reshape(W(f,z,:),T,1);
        cv = conv(v,H(z,:));
        Lambda{z}(f,:) = Lambda{z}(f,:) + cv(1:N);
    end
end


LambdaTot = zeros(M,N);
for z = 1:R
    LambdaTot = LambdaTot +Lambda{z};
end

for z = 1:R
    xs{z} = istft(sp.*Lambda{z}./LambdaTot,Nfft,hamming(Nfft,'periodic')',Nfft/4);
    soundsc(xs{z},sr);
end




%% plot results
close all


d = length(s)/sr;

figure 
imagesc((1:N)/N*d,0:0.05:sr/2000,db(V))
title('Original Spectrogram')
xlabel('time (s)')
ylabel('frequency (kHz)')

axis xy
mx = max(caxis);
caxis([mx-120,mx])
dyn = caxis;

figure
subplot(131)
imagesc((1:T)/N*d,0:0.05:sr/2000,db(reshape(W(:,1,:),M,T)))
caxis(dyn)
axis xy
title('First template')
xlabel('time (s)')
ylabel('frequency (kHz)')

subplot(132)
imagesc((1:T)/N*d,0:0.05:sr/2000,db(reshape(W(:,2,:),M,T)))
caxis(dyn)
axis xy
title('Second template')
xlabel('time (s)')
ylabel('frequency (kHz)')

subplot(133)
imagesc((1:T)/N*d,0:0.05:sr/2000,db(reshape(W(:,3,:),M,T)))
caxis(dyn)
axis xy
title('Third template')
xlabel('time (s)')
ylabel('frequency (kHz)')

figure
subplot(311)
plot((1:N)/N*d,(H(1,:)))
title('Activation of the 1st template')
xlabel('time (s)')
subplot(312)
plot((1:N)/N*d,(H(2,:)))
title('Activation of the 2nd template')
xlabel('time (s)')
subplot(313)
plot((1:N)/N*d,(H(3,:)))
title('Activation of the 3rd template')
xlabel('time (s)')

