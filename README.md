# NMF-matlab

Matlab code of Non-Negative matrix factorization (NMF) and variants, using multiplicative update rules for a beta-divergence cost (including Itakura Saito divergence, Kullback Leibler divergence and Froebenius distance).

Implemented variants:
- Probabilistic Latent Component Analysis (PLCA) as proposed by Smaragdis. This is actually a probabilistic model for NMF that corresponds to a Kullback-Leibler cost.
- Non Negative Factor Deconvolution (NMFD) as proposed by Smaragdis. This is a time-convolutive variant of NMF that factorize time/frequency patterns. Useful for instance to model drum sounds.
- Non Negative Matrix Factor 2-D Deconvolution (NMF2D) as proposed in Non Negative Matrix Factor 2-D Deconvolution for Blind Single Channel Source Separation by Mikkel N. Schmidt and Morten Mørup. This is a time-frequency-convolutive variant of NMF that factorize time/frequency patterns that can be shifted both in time and frequency. Useful to decompose representation for which frequency shift make sense, such as Constant-Q spectrogram.