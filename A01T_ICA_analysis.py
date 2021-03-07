
#########################################################
# This is the code for observing and studying the raw data
#coming from EEG signals. Authors of this code are
#Kirtan Jha and Naren Vishwanath Kalburgi
#We would like to give credits to online documentation of MNE software 
#through which we learnt to use it in python. 
########################################################

#To study raw signals first there are some dependencies in python which needs to be installed and imported
import numpy as np
import matplotlib.pyplot as plt
from time import time
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout, read_custom_montage, find_layout, make_standard_montage, make_eeg_layout
from mne.io import concatenate_raws, read_raw_gdf, read_raw_eeglab
from mne.preprocessing import ICA


# In[45]:
#Importing the raw signal from the local computer

fileloc = 'C:\\Users\\harsh\\Documents\\MATLAB\\A01T.gdf' # ME-graz-block.gdf , ME-graz-rand.gdf_proc.set

raw = read_raw_gdf(fileloc, stim_channel=-1, preload=True) #.gdf


# In[41]: This will plot every channel's signal 

raw.plot()


# In[25]: 


ch_names = raw.ch_names
print(ch_names)
#montage = read_custom_montage('chanloc.loc') coord_frame='head')#32ch
montage = make_standard_montage('biosemi32')
montage.plot(kind='topomap', show_names=True)

ch_loc = montage.ch_names
print(ch_loc)
c = dict(zip(ch_names, ch_loc)) #combine names with locations
raw.rename_channels(c)
raw.set_montage(montage)
fig = montage.plot(kind='3d')
fig.gca().view_init(azim=25, elev=20)


# In[32]:


def plot_two_chans(ch1, ch2, title):
  channel_names = [ch1, ch2]
  two_eeg_chans = raw[channel_names, 500:5000]
  y_offset = np.array([5e-10, 0.001])  # just enough to separate the channel traces
  x = two_eeg_chans[1]
  y = two_eeg_chans[0].T + y_offset
  lines = plt.plot(x, y)
  plt.legend(lines, channel_names)
  plt.title(title)
  plt.show()
plot_two_chans('C3','C4', 'Raw EEG')


# In[35]:


raw.copy().pick_types(stim=True).plot(start=3, duration=6)


# In[23]: Applying bandpass filter to check how it works in the main code. Here we have applied to only 2 channels. 


lo_freq = 1.
hi_freq = 50.

raw.filter(lo_freq, hi_freq, fir_design='firwin', skip_by_annotation='edge')

plot_two_chans('C3','C4', 'band-passed')


# In[17]: Averaging 


raw.set_eeg_reference(ref_channels='average', projection=True)
plot_two_chans('C3','C4', 'Average reference')


# In[38]: event_ids are defined in dataset description which is 769,770,771,772 
#equivalent to 7,8,9,10


tmin, tmax = 3,6 .
event_id = dict(left=7, right=8, foot=9,tongue=10) 
events, _ = events_from_annotations(raw, event_id='auto')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 4s)
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
epochs_train = epochs.copy().crop(tmin=3., tmax=6.)
labels = epochs.events[:, -1] - 2


# In[19]: starting independent component analysis


ica = ICA(n_components=0.95, method='fastica').fit(epochs)

ica.plot_components() # plot components

ica.plot_properties(epochs) # Display component properties.


# In[20]:


ica.exclude = [0, 1]  # indices chosen based on various plots above

# ica.apply() changes the Raw object in-place, so let's make a copy first:
reconst_raw = raw.copy()
ica.apply(reconst_raw)

#raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
#reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
#del reconst_raw


# In[21]:



ica.plot_sources(raw)


# In[22]: 


from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test as pcluster_test
from mne.viz.utils import center_cmap
picks = mne.pick_channels(raw.info["ch_names"], ["C3", "Cz", "C4"])
epochs = Epochs(raw, events, event_id, tmin - 0.5, tmax + 0.5,
                    picks=picks, baseline=None, preload=True)
# compute ERDS maps 
freqs = np.arange(7, 30, 1)  # frequencies from 7-30Hz
n_cycles = freqs  # use constant t/f resolution
vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
baseline = [-1, 0]  # baseline interval (in s)
cmap = center_cmap(plt.cm.RdBu, vmin, vmax)  # zero maps to white
kwargs = dict(n_permutations=100, step_down_p=0.05, seed=1,
              buffer_size=None)  # for cluster test

# Run timefrequency decomposition overall epochs
tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                     use_fft=True, return_itc=False, average=False,
                     decim=2)
tfr.crop(tmin, tmax)
tfr.apply_baseline(baseline, mode="percent")
for event in event_id:
    # select desired epochs for visualization
    tfr_ev = tfr[event]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [10, 10, 10, 1]})
    for ch, ax in enumerate(axes[:-1]):  # for each channel
        # positive clusters
        _, c1, p1, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=1, **kwargs)
        # negative clusters
        _, c2, p2, _ = pcluster_test(tfr_ev.data[:, ch, ...], tail=-1,
                                     **kwargs)

        # note that we keep clusters with p <= 0.05 from the combined clusters
        # of two independent tests; in this example, we do not correct for
        # these two comparisons
        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)

        # plot TFR (ERDS map with masking)
        tfr_ev.average().plot([ch], vmin=vmin, vmax=vmax, cmap=(cmap, False),
                              axes=ax, colorbar=False, show=False, mask=mask,
                              mask_style="mask")

        ax.set_title(epochs.ch_names[ch], fontsize=10)
        ax.axvline(0, linewidth=1, color="black", linestyle=":")  # event
        if not ax.is_first_col():
            ax.set_ylabel("")
            ax.set_yticklabels("")
    fig.colorbar(axes[0].images[-1], cax=axes[-1])
    fig.suptitle("ERDS ({})".format(event))
    fig.show()


# In[26]:


