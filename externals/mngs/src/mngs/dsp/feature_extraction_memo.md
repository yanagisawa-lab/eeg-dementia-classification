"""

number of crossing

number of cwt peaks

binned entropy
fourier entropy

dickey fuller

MuLawEncoding
    waveform, sample_rate = torchaudio.load('test.wav', normalize=True)
    transform = torchaudio.transforms.MuLawEncoding(quantization_channels=512)
    mulawtrans = transform(waveform)

FrequencyMasking
    torchaudio.transforms.FrequencyMasking(freq_mask_param: int, iid_masks: bool = False)

## Feature extractions
https://pytorch.org/audio/stable/transforms.html#feature-extractions
from torchaudio import transforms
waveform, sample_rate = torch.randn(batch, 1024), 250
waveform = waveform.cuda()

Spectrogram
    <!-- waveform, sample_rate = torchaudio.load('test.wav', normalize=True) -->
    transform = torchaudio.transforms.Spectrogram(n_fft=sample_rate).cuda()
    spectrogram = transform(waveform)
    print(spectrogram.shape) # [2,126,9]

<!-- GriffinLim
 !--     batch, freq, time = 2, 257, 100
 !--     spectrogram = torch.randn(batch, freq, time)
 !--     transform = transforms.GriffinLim(n_fft=512)
 !--     waveform = transform(spectrogram) -->

torchaudio.transforms.ComputeDeltas
    <!-- waveform, sample_rate = torchaudio.load('test.wav', normalize=True) -->
    <!-- waveform, sample_rate = torch.randn(batch, 1024), 256 -->
    transform = transforms.PitchShift(sample_rate, 4).cuda()
    waveform_shift = transform(waveform)  # (channel, time)
    waveform_shift_sum = waveform_shift.sum(dim=-1)
    print(waveform_shift_sum)

torchaudio.transforms.SpectralCentroid
    <!-- waveform, sample_rate = torchaudio.load('test.wav', normalize=True) -->
    transform = transforms.SpectralCentroid(sample_rate).cuda()
    spectral_centroid = transform(waveform)  # (channel, time)
    print(spectral_centroid)

torchaudio.transforms.RNNTLoss
    logits = torch.tensor([[[[0.1, 0.6, 0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.6, 0.1, 0.1],
                             [0.1, 0.1, 0.2, 0.8, 0.1]],
                            [[0.1, 0.6, 0.1, 0.1, 0.1],
                             [0.1, 0.1, 0.2, 0.1, 0.1],
                             [0.7, 0.1, 0.2, 0.1, 0.1]]]],
                          dtype=torch.float32,
                          requires_grad=True)
    targets = torch.tensor([[1, 2]], dtype=torch.int)
    logit_lengths = torch.tensor([2], dtype=torch.int)
    target_lengths = torch.tensor([2], dtype=torch.int)
    transform = transforms.RNNTLoss(blank=0)
    loss = transform(logits, targets, logit_lengths, target_lengths)
    loss.backward()

## Multi-channel
https://pytorch.org/audio/stable/transforms.html#multi-channel

n_chs = 160
multi_channel_waveform = torch.randn(batch, n_chs, time).cuda()

PSD
    transform = torchaudio.transforms.PSD(multi_mask= False,
                                          normalize= True,
                                          eps = 1e-15).cuda()


    specgram =
    psd = transform(waveform)  # (channel, time)
    print(spectral_centroid)





https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html
agg_autocorrelation
approximate_entropy
augmented_dickey_fuller
autocorrelation
benford_correlation
binned_entropy
cwt_coefficients
fft_aggregated
fft_coefficient
fourier_entropy
matrix_profile
number_crossing_m
number_cwt_peaks
number_peaks
partial_autocorrelation
permutation_entropy
quantile
ratio_beyond_r_sigma
root_mean_square
sample_entropy
skewness

- spkt_weilch_density
  This feature calculator estimates the cross power spectral density of the time series x at different frequencies.


- stadard_deviation

- sum_values

- variance

-
"""
