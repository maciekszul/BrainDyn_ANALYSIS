from nilearn import image, input_data, masking, plotting, signal
from nilearn.datasets import MNI152_FILE_PATH
from nilearn.plotting import plot_roi, plot_epi, show
import numpy as np
import matplotlib.pylab as plt
from scipy.fftpack import fft, fftfreq
from scipy.signal import detrend, resample
from utilities import files
import os.path as op


path = "/home/mszul/datasets/braindyn_bids/derivatives/fmriprep"
participants = files.get_folders_files(path)[0]
participant = participants[0]
_id = participant.split("/")[-1]


mask = "masks/hocp-Occipital_Pole.nii.gz"
nifti = op.join(
    participant, 
    "func", 
    "{}_task-faces_dir-AP_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz".format(_id)
)
brain_mask = op.join(
    participant,
    "func",
    "{}_task-flicker_dir-AP_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz".format(_id)
)
t1 = op.join(
    participant,
    "anat",
    "{}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz".format(_id)
)

flicker = image.smooth_img(nifti, fwhm=None)
ffg_mask = image.smooth_img(mask, fwhm=None)
ffg_mask = image.resample_to_img(
    ffg_mask, flicker, fill_value=0, 
    clip=True, interpolation="nearest"
)

data = masking.apply_mask(flicker, ffg_mask, smoothing_fwhm=3)

data = signal.clean(data, standardize=False, detrend=True)

og_time = data.shape[0]*1.3
old_t = np.arange(data.shape[0] * 1.3)

zscored_map = []
# for row in range(2):
for row in range(data.shape[1]):
    res_y, res_t = resample(data[:,row], int(og_time/0.0005), t=old_t, axis=0)
    fft_Y = fft(res_y, axis=0)
    fft_FREQ = fftfreq(fft_Y.shape[0], 0.0005)
    fft_Y = fft_Y[:200]
    fft_FREQ = fft_FREQ[:200]

    below = np.where((fft_FREQ > 0.01) & (fft_FREQ < 0.035))[0]
    above = np.where((fft_FREQ > 0.040) & (fft_FREQ < 0.067))[0]
    target_freq = np.where((fft_FREQ >= 0.035) & (fft_FREQ <= 0.040))[0]
    around = np.concatenate([above, below])
    around.sort()

    selecta = np.take(fft_Y, around, axis=0)
    target = np.take(fft_Y, target_freq, axis=0)
    target_zsc = np.real(np.apply_along_axis(lambda x: (x - selecta.mean())/selecta.std(), 0, target))
    zsc = max(target_zsc.min(), target_zsc.max(), key=abs)
    zscored_map.append(zsc)
    print(row, "/", data.shape[1])

zscored_map = np.array(zscored_map)
np.save("occipital-lobe.npy", zscored_map)