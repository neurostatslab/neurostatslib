---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Hippocampal place cell sequences

```{code-cell} ipython3
:tags: [hide-cell]

# suppress warnings
import warnings
warnings.simplefilter("ignore")

# imports
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import signal
import seaborn as sns
import tqdm
import pynapple as nap
import pynacollada as nac
from scipy.ndimage import gaussian_filter1d
import textwrap

# necessary for animation
import nemos as nmo
plt.style.use(nmo.styles.plot_style)

# define helper functions
def printeval(expr):
    print(expr, " = \n", textwrap.indent(eval(expr).__str__(), 4*' '),"\n")

def plot_place_fields(pf, axs, title=None, xlabel=None):
    for c, ax in zip(pf, axs):
        ax.fill_between(pf[c].index.values, np.zeros_like(pf[c]), pf[c].values)
        ax.set_yticks([])
        # ax.set_xticks([])
    if title is not None:
        axs[0].set_title(title)
    if xlabel is not None:
        axs[-1].set_xlabel(xlabel)
```

## Background

+++

## Exploring the data
    
The data set we'll be looking at is from the manuscript [Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences](https://www.science.org/doi/10.1126/science.aad1935). In this study, the authors collected electrophisiology data in rats across multiple sites in layer CA1 of hippocampus to extract the LFP alongside spiking activity of many simultaneous pyramidal units. In each recording session, data were collected while the rats explored a novel environment (a linear or circular track), as well as during sleep before and after exploration. In our following analyses, we'll focus on the exploration period of a single rat and recording session.

The full dataset for this study can be accessed on [DANDI](https://dandiarchive.org/dandiset/000044/0.210812.1516). Since the file size of a recording session can be large from the LFP saved for each recorded channel, we'll use a smaller file that contains the spiking activity and the LFP from a single, representative channel, which is hosted on [OSF](https://osf.io/2dfvp). This smaller file, like the original data, is saved as an [NWB](https://www.nwb.org) file.

For this tutorial, we can use the pynacollada function `load_data` and the tutorial name `"place_cells"` to fetch the data. Under the hood, this function uses pynapple to load in the NWB file, which returns a dictionary of pynapple objects that have been extracted from the file. The next sections will explore each of these fields (excluding `"theta_phase"`, which we'll compute ourselves later on).

```{code-cell} ipython3
# load pynacollada data set
data = nac.load_data("place_cells")
print(data)
```

### `units` 
    
The `units` field is a [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html#pynapple.TsGroup): a collection of [`Ts`](https://pynapple.org/generated/pynapple.Ts.html) objects containing the spike times of each unit, where the "Index" is the unit number or key. Each unit has the following [metadata](https://pynapple.org/user_guide/03_metadata.html):
- **rate**: computed by pynapple, is the average firing rate of the neuron across all recorded time points.
- **location**, **shank**, and **cell_type**: variables saved and imported from the original data set.

```{code-cell} ipython3
printeval("data['units']")    # print the entire TsGroup object
printeval("data['units'][1]") # print the Ts object corresponding to unit 1
```

### `rem`, `nrem`, and `forward_ep`


The next three objects; `rem`, `nrem`, and `forward_ep`; are all [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html#pynapple.IntervalSet) objects containing time windows of REM sleep, nREM sleep, and forward runs down the linear maze, respectively. All intervals in `forward_ep` occur in the middle of the session, while `rem` and `nrem` both contain sleep epochs that occur before and after exploration. 

```{code-cell} ipython3
printeval("data['rem']")            # print the REM intervals
printeval("data['nrem']")           # print the nREM intervals
printeval("data['forward_ep']")     # print the forward_ep intervals

# grab the first time stamp to plot relative measure of time within session
t_start = data["nrem"].start[0]

fig,ax = plt.subplots(figsize=(10,2), constrained_layout=True)

# plot blue rectangles for each "rem" interval
sp1 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="blue", alpha=0.1) for iset in data["rem"]];
# plot green rectangles for each "nrem" interval
sp2 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="green", alpha=0.1) for iset in data["nrem"]];
# plot red rectangles for each "forward_ep" interval
sp3 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="red", alpha=0.1) for iset in data["forward_ep"]];

ax.set(xlabel="Time within session (minutes)", title="Labelled time intervals across session", yticks=[])
ax.legend([sp1[0],sp2[0],sp3[0]], ["REM sleep","nREM sleep","forward runs"]);
```

### `eeg`


The `eeg` field is a [`TsdFrame`](https://pynapple.org/generated/pynapple.TsdFrame.html) containing an LFP voltage trace for a single representative channel in CA1.

```{code-cell} ipython3
printeval("data['eeg']")       # print the 2D TsdFrame
printeval("data['eeg'][:,0]")  # slice the first column, printing a 1D Tsd
```

### `position`
The final object, `position`, is a [`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html) containing the linearized position of the animal, in centimeters, recorded during the exploration window. This object has the field `time_support`, which is the `IntervalSet` during which position is recorded, i.e. the time window spanning the exploration period. When the animal is at rest, the position is recorded as `nan`, which means "not a number".

```{code-cell} ipython3
printeval("data['position']")               # print the position Tsd
printeval("data['position'].time_support")  # print the position's time support

# this plot demonstrates that "forward_ep" corresponds to positively-increasing position
pos_start = data["position"].time_support.start[0]  # grab the start of the exploration window
fig, ax = plt.subplots(figsize=(10,3))
l1 = ax.plot(data["position"]) # plot position, where nan values are blank
# overlay forward_ep windows
l2 = [ax.axvspan(iset.start[0], iset.end[0], color="red", alpha=0.1) for iset in data["forward_ep"]];
# set the x-limits to the first 300 seconds, set other labels
ax.set(xlim=[pos_start,pos_start+300], ylabel="Position (cm)", xlabel="Time (s)", title="Tracked position along linear maze")
ax.legend([l1[0], l2[0]], ["animal position", "forward run epochs"]);
```

## Identifying place fields
Our first analysis will identify each unit's place field. We can do this using the pynapple function `compute_1d_tuning_curves` and then apply some post-processing to clean up and isolate units with strong spatial selectivity. Specifically, the following analysis will:
1. Extract the relevant fields from the loaded dictionary
2. Compute the tuning curves, i.e. place fields, using pynapple
3. Apply a Gaussian filter to smooth the place fields
4. Filter for "good" place cells by finding units with high spatial selectivity
6. Sort the units in order of spatial preference
7. Plot the resulting place fields

```{code-cell} ipython3
##-- 1. Grab the data --##
spikes = data["units"]                  # grab all the units
forward_ep = data["forward_ep"]         # grab the forward run epochs
position = data["position"].dropna()    # grab position, and drop nan values
lfp = data["eeg"][:,0]                  # grab the first eeg column

##-- 2. Compute place fields --##
# compute place fields in 2cm bins across linear track
track_len = position.max() - position.min() # get total length of the track (in cm)
nb_bins = int(track_len / 2)                # number of 2 cm bins, rounded to an integer
# use pynapple to compute 1d tuning curves over position, returning a pandas dataframe
place_fields = nap.compute_1d_tuning_curves(spikes, position, nb_bins = nb_bins)

##-- 3. Smooth place fields --##
# use scipy to smooth the place fields
sigma = 2.5 # we want a 5cm Gaussian kernel, which is 5cm / 2cm/bin = 2.5 bins
place_fields[:] = gaussian_filter1d(place_fields.values, sigma, axis=0) # update in place

##-- 4. Filter units to "good" place cells --##
# find all points at which the firing rate is 2 s.d. greater than the mean
rate_95p = place_fields > (place_fields.mean() + 2*place_fields.std())
# find whether there are 5 consecutive bins of high firing
# we can do this by taking a rolling sum of width 5 on the boolean array, and check whether any sum is 5
good_idx = np.any(rate_95p.rolling(5).sum() == 5, axis=0) 
# we also want the firing peak to be at least 1 Hz, and the average rate less than a theta-modulated interneuron
good_idx = good_idx & (place_fields.max() >= 1) & (spikes.rate < 6)
# filter the place fields using good_idx
place_fields = place_fields.loc[:,good_idx] 

##-- 5. Sort units in order of preferred location --#
# idxmax will get the index of max firing, argsort will return the index that sorts idxmax
sort_order = place_fields.idxmax().argsort() 
sorted_place_fields = place_fields.iloc[:, sort_order] # sort units, which are the columns

##-- 6. Plot results --##
fig, axs = plt.subplots(len(place_fields.columns), 1, sharex=True, figsize=(3,8))
plot_place_fields(sorted_place_fields, axs, title="Sorted place fields", xlabel="Position (cm)")
```

## Identifying place cell remapping

The previous analysis computed place fields across all positions. However, place cell encoding is often selective to the direction the animal is moving along a linear maze, where place cells remap depending on which way the animal running. This means place cells can encode different locations depending on whether the animal is running forwards or backwards, leading to separable place field sequences that encode each direction. We can show this by:
1. Compute place fields on forward runs only
2. Compute place fields on backward runs only
3. Plot "forward" and "backward" fields, sorted by "forward" preference
4. Plot "forward" and "backward" fields, sorted by "backward" preference

What we will see is that forward fields form a sequence when sorted by forward preference, but not backward fields; the opposite will be true when sorting by backward preference.

```{code-cell} ipython3
# filter spike times to good units calculated above
good_spikes = spikes[good_idx]

##-- 1. Compute place fields on forward runs --##
# now we pass the argument ep=forward_ep to specify that tuning curves are only calculated during forward runs
forward_fields = nap.compute_1d_tuning_curves(good_spikes, position, ep=forward_ep, nb_bins=nb_bins)
forward_fields[:] = gaussian_filter1d(forward_fields.values, sigma, axis=0)

##-- 2. Compute place fields on backward runs --##
# all running occurs when there is a non-NaN position
run_ep = position.time_support
# backward runs are the set difference between all runs and forward runs
backward_ep = run_ep.set_diff(forward_ep)
# pass ep=backward_run to compute only during backward runs
backward_fields = nap.compute_1d_tuning_curves(good_spikes, position, ep=backward_ep, nb_bins=nb_bins)
backward_fields[:] = gaussian_filter1d(backward_fields.values, sigma, axis=0)

# plot results
fig = plt.figure(figsize=(7,8))
# 3 columns, with the middle being empty for visual separation
subfigs = fig.subfigures(1,3, width_ratios=[2,0.1,2]) 

##-- 3. Plot forward and backward place fields sorted by forward preference --##
# split left column into two subcolumns for each direction
axs = subfigs[0].subplots(len(forward_fields.columns), 2, sharex=True)
# get sort order of forward fields
sort_order = forward_fields.idxmax().argsort()
# plot forward fields sorted by forward preference in left subcolumn
plot_place_fields(forward_fields.iloc[:,sort_order], axs[:,0], "Forward runs", "Position (cm)")
# plot backward fields sorted by forward preference in right subcolumn
plot_place_fields(backward_fields.iloc[:,sort_order], axs[:,1], "Backward runs", "Position (cm)")
subfigs[0].suptitle("Place fields sorted by forward runs")

## 4. Plot forward and backward place fields sorted by backward preference --##
# split right column into two subcolumns
axs = subfigs[2].subplots(len(forward_fields.columns), 2, sharex=True)
# get sort order of backward fields
sort_order = backward_fields.idxmax().argsort()
# plot forward fields sorted by backward preference in left subcolumn
plot_place_fields(forward_fields.iloc[:,sort_order], axs[:,0], "Forward runs", "Position (cm)")
# plot backward fields sorted by backward preference in right subcolumn
plot_place_fields(backward_fields.iloc[:,sort_order], axs[:,1], "Backward runs", "Position (cm)")
subfigs[2].suptitle("Place fields sorted by backward runs");
```

## Visualizing LFP
While the animal is running, we can expect the LFP to be dominated by 6-12 Hz theta oscillations. We can visualize theta power over time by using the pynapple function `compute_wavelet_transform`, which computes a continuous wavelet transform over some input time series.

A [continuous wavelet transform](https://en.wikipedia.org/wiki/Continuous_wavelet_transform) decomposes a signal into a set of [wavelets](https://en.wikipedia.org/wiki/Wavelet), in this case [Morlet wavelets](https://en.wikipedia.org/wiki/Morlet_wavelet), that span both frequency and time. You can think of the wavelet transform as a cross-correlation between the signal and each wavelet, giving the similarity between the signal and various frequency components at each time point of the signal. Similar to a Fourier transform, this gives us an estimate of what frequencies are dominating a signal. Unlike the Fourier tranform, however, the wavelet transform gives us this estimate as a function of time.

The following example will:
1. Restrict the data to an example run
2. Compute the wavelet transform using pynapple
3. Plot the raw LFP, the animal position, and the wavelet transform results

```{code-cell} ipython3
##-- 1. Restrict data to example forward run --##
# grab a single run and add two seconds to the end
ex_run_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end+2)
# restrict lfp and position to example run
ex_lfp_run = lfp.restrict(ex_run_ep)
ex_position = position.restrict(ex_run_ep)

##-- 2. Compute wavelet transform --##
# compute at 100 log-spaced samples between 5Hz and 200Hz
freqs = np.geomspace(5, 200, 100)
sample_rate = 1250 # known from manuscript methods
# use pynapple to compute the wavelet transform, supplying a known sampling rate
cwt_run = nap.compute_wavelet_transform(ex_lfp_run, freqs, fs=sample_rate)

##-- 3. Plot results --##
# 2 rows, 1 column, top plot smaller than bottom
fig, axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, height_ratios=[0.3, 1.0], sharex=True)

# plot the raw LFP on the top plot
p1 = axs[0].plot(ex_lfp_run)
axs[0].set(title="Example run", ylabel="LFP (a.u.)")
# axs[0].margins(0)
# plot the animal position on the top plot using a separate y-axis
ax = axs[0].twinx()
p2 = ax.plot(ex_position, color="orange")
ax.set_ylabel("Position (cm)")
ax.legend([p1[0], p2[0]],["raw LFP","animal position"])

# plot the wavelet transform on the bottom
amp = np.abs(cwt_run.values) # plot the amplitude of the transform, which it its absolute value
cax = axs[1].pcolormesh(cwt_run.t, freqs, amp.T, cmap="magma")
axs[1].set(title="Wavelet decomposition", ylabel="Frequency (Hz)", xlabel="Time(s)", yscale='log', yticks=freqs[::10], yticklabels=np.rint(freqs[::10]));
axs[1].minorticks_off()
fig.colorbar(cax,label="Amplitude",location="right", pad=-0.04);
```

## Filtering for theta
We can isolate the theta frequency band by applying a [bandpass filter](https://en.wikipedia.org/wiki/Band-pass_filter) to the LFP trace. We can do this with the pynapple function `apply_bandpass_filter`.

```{code-cell} ipython3
# redefine example run to run only
ex_run_ep = forward_ep[9]

# use pynapple to filter the LFP between 6 and 12 Hz, supplying a known sampling rate
theta_band = nap.apply_bandpass_filter(lfp, (6.0, 12.0), fs=sample_rate)

# plot results
plt.figure(constrained_layout=True, figsize=(10, 3))
# plot raw lfp restricted to example run
plt.plot(lfp.restrict(ex_run_ep), label="raw")
# plot filtered theta band restricted to example run
plt.plot(theta_band.restrict(ex_run_ep), label="filtered")
plt.xlabel("Time (s)")
plt.ylabel("LFP (a.u.)")
plt.title("Bandpass filter for theta oscillations (6-12 Hz)")
plt.legend();
```

## Computing theta phase
In order to examine phase precession in place cells, we need to identify the instantaneous phase of theta. We can do this by taking the angle of the [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform) of the band-pass filtered signal.

```{code-cell} ipython3
# compute phase using the angle of the hilbert transform of the filtered signal
phase = np.angle(signal.hilbert(theta_band)) 
phase = (phase + 2 * np.pi) % (2 * np.pi) # wrap to [0,2pi]
# store phase into a pynapple object for easy time alignment
theta_phase = nap.Tsd(t=theta_band.t, d=phase)

# plot results
fig,ax = plt.subplots(figsize=(10,2), constrained_layout=True) #, sharex=True, height_ratios=[2,1])
# plot theta phase restricted to example run
p1 = ax.plot(theta_phase.restrict(ex_run_ep), color='r')
ax.set(ylabel="Phase (rad)", xlabel="Time (s)")
# plot filtered LFP restricted to example run on a separate y-axis
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_ep))
ax.set_ylabel("LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```

## Computing phase precession
Phase precession is captured by the relationship between the *position* at which a place field spikes and the corresponding *phase* of theta. Specifically, there is a negative relationship: at *early positions*, cells will fire at *late phases* of theta, and at *late positions*, cells will fire at *early phases* of theta. The following example demonstrates how to find the position and phase at which an example unit fires, as well as plot multiple visualizations to demonstrate phase precession within a single run and across all sessions.

```{code-cell} ipython3
unit = 177 # example unit with place field in the center

##-- Use pynapple value_from to grab various values corresponding to each spike time --##
spike_theta = spikes[unit].value_from(theta_band)  # filtered LFP value closest to each spike
spike_phase = spikes[unit].value_from(theta_phase) # theta phase closest to each spike
spike_position = spikes[unit].value_from(position) # position closest to each spike

##-- Plotting --##
# split figure into two subfigures
fig = plt.figure(figsize=(10,5), constrained_layout=True)
subfigs = fig.subfigures(1,2)

### in left subfigure, plot data restricted to the example run
subfigs[0].suptitle("Phase precession within a single forward run")
axs = subfigs[0].subplots(3, 1, sharex=True) # 3 subplots

## left top plot - spikes vs LFP, shows a "U" of spike times relative to LFP
# plot raw LFP
axs[0].plot(lfp.restrict(ex_run_ep), alpha=0.5, label="raw LFP")
# plot filtered LFP
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
# plot the value of the filtered LFP at each spike time of the example unit
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)", title="Spike times relative to LFP")
axs[0].legend(loc="center left")

## left middle plot - spikes vs phase, shows a negative line of spike times relative to phase
# plot the phase of theta
axs[1].plot(theta_phase.restrict(ex_run_ep), color="slateblue", label="theta phase")
# plot the value of the phase at each spike time of the example unit
axs[1].plot(spike_phase.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[1].set(ylabel="Phase (rad)", title="Spike times relative to theta phase")
axs[1].legend(loc="center left")

## left bottom plot - position, shows where animal is on the track and when its in the unit's place field
ex_position = position.restrict(ex_run_ep) # store restricted result because we use it more than once
# plot example position as a dashed line
axs[2].plot(ex_position, '--', color="green", label="animal position")
# plot approximate field bounds in solid line
axs[2].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[2].set(ylabel="Position (cm)", xlabel="Time (s)", title="Animal position")
axs[2].legend()

### in right subfigure, plot session-wide visualizations
subfigs[1].suptitle("Phase precession across all forward runs")
axs = subfigs[1].subplots(2,1, height_ratios=[2,1], sharex=True) # 2 subplots
## top right plot - phase precession scatter plot, position vs phase
# for each spike time, plot its corresponding position and theta phase
axs[0].plot(spike_position.restrict(forward_ep), spike_phase.restrict(forward_ep), 'o')
axs[0].set(ylabel="Phase (rad)", title="Phase x position of each spike time")
## bottom right plot - example unit place field
pf = forward_fields[unit]
axs[1].fill_between(pf.index.values, np.zeros_like(pf), pf.values)
axs[1].set(title=f"Unit {unit} place field", ylabel="Firing rate (Hz)", xlabel="Position (cm)");
```

## Bayesian decoding of position
A common analysis applied to hippocampal data is [Bayesian decoding](https://pubmed.ncbi.nlm.nih.gov/9463459/) to predict the position an animal is likely at given current spiking activity. 

### Background

Recall Bayes' rule, written here in terms of our relevant variables:

$$P(position|spikes) = \frac{P(position)P(spikes|position)}{P(spikes)}$$

Our goal is to compute the unknown posterior $P(position|spikes)$ given known prior $P(position)$ and known likelihood $P(spikes|position)$. 

$P(position)$, also known as the *occupancy*, is the probability that the animal is occupying some position. This can be computed exactly by the proportion of the total time spent at each position, but in many cases it is sufficient to estimate the occupancy as a uniform distribution, i.e. it is equally likely for the animal to occupy any location.

The next term, $P(spikes|position)$, which is the probability of seeing some sequence of spikes across all neurons at some position. Computing this relys on the following assumptions:
1. Neurons fire according to a Poisson process (i.e. their spiking activity follows a Poisson distribution)
2. Neurons fire independently from one another.

While neither of these assumptions are strictly true, they are generally reasonable for pyramidal cells in hippocampus and allow us to simplify our computation of $P(spikes|position)$

The first assumption gives us an equation for $P(spikes|position)$ for a single neuron, which we'll call $P(spikes_i|position)$ to differentiate it from $P(spikes|position) = P(spikes_1,spikes_2,...,spikes_i,...,spikes_N|position) $, or the total probability across all $N$ neurons. The equation we get is that of the Poisson distribution:
$$
P(spikes_i|position) = \frac{(\tau f_i(position))^n e^{-\tau f_i(position)}}{n!}
$$
where $f_i(position)$ is the firing rate of the neuron at position $(position)$ (i.e. the tuning curve), $\tau$ is the width of the time window over which we're computing the probability, and $n$ is the total number of times the neuron spiked in the time window of interest.

The second assumptions allows us to simply combine the probabilities of individual neurons. Recall the product rule for independent events: $P(A,B) = P(A)P(B)$ if $A$ and $B$ are independent. Treating neurons as independent, then, gives us the following:
$$
P(spikes|position) = \prod_i P(spikes_i|position)
$$

The final term, $P(spikes)$, is inferred indirectly using the law of total probability:

$$P(spikes) = \sum_{position}P(position,spikes) = \sum_{position}P(position)P(spikes|position)$$

Another way of putting it is $P(spikes)$ is the normalization factor such that $\sum_{position} P(position|spikes) = 1$, which is achived by dividing the numerator by its sum.

<div class="alert alert-block alert-info">
<b>Important:</b> Generally this method is cross-validated, which means you train the model on one set of data and test the model on a different, held-out data set. For Bayesian decoding, the "model" refers to the model *likelihood*, which is computed from the tuning curves.
</div>

### Running the analysis
We can perform Baysian decoding of position using the pynapple function [`decode_1d`](https://pynapple.org/generated/pynapple.process.decoding.html#pynapple.process.decoding.decode_1d). The following example will:
1. Cross-validation: Split the data into a training set for rate map computation, and a test set for decoding
2. Compute tuning curves during the training set
3. Smooth the spike counts for a better decoder estimate
4. Decode position using pynapple
5. Plot the results

```{code-cell} ipython3
##-- 1. Cross-validation setup --##
# test set as an example trial
ep_test = forward_ep[9]
# hold out trial from place field computation
ep_train = forward_ep.set_diff(ex_run_ep)

##-- 2. Compute place fields --##
# compute place fields during training set and smooth
place_fields = nap.compute_1d_tuning_curves(spikes, position, ep=ep_train, nb_bins=nb_bins)
place_fields[:] = gaussian_filter1d(place_fields.values, sigma, axis=0)

##-- 3. Smooth spike counts --##
# restrict all spikes to test epoch and count in 40ms bins
counts = spikes.restrict(ep_test).count(0.04)
# smooth counts by convolving with a length-5 window of ones (i.e. moving sum of length 5)
# this will make bins be 40ms * 5 = 200ms wide
smth_counts = counts.convolve(np.ones(5))

##-- 4. Decode position with pynapple --##
# decode over ep_test, giving the smoothed spike counts
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ep_test, bin_size=0.2)

##-- 5. Plot the results --##
fig,ax = plt.subplots(figsize=(10, 3), constrained_layout=True)
# plot the decoded probability of position
c = ax.pcolormesh(smth_decoded_position.index,place_fields.index,np.transpose(smth_decoded_prob), cmap="magma")
# plot the decoded position, i.e. the position of max probability, as a dashed green line
ax.plot(smth_decoded_position, "--", color="limegreen", label="decoded position")
# plot the true animal position as a solid green line
ax.plot(position.restrict(ex_run_ep), color="limegreen", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)");
```

## Visualization of theta sequences

Theta sequences are often visualized by using Bayesian decoding on a faster time scale. The following example will repeat the steps from the previous analysis, except now predicting spiking activity in shorter time bins: specifically, during 50 ms smoothed bins.

```{code-cell} ipython3
##-- Smooth counts in smaller time windows --##
counts = spikes.restrict(ep_test).count(0.01) # count in 10ms bins
smth_counts = counts.convolve(np.ones(5))     # moving sum of length 5

##-- Run decoding on test epoch --##
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ep_test, bin_size=0.05)

##-- Plot the decoded probability, predicted position, and true position --##
fig, axs = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True, height_ratios=[3,1], sharex=True)
c = axs[0].pcolormesh(smth_decoded_prob.index, smth_decoded_prob.columns, np.transpose(smth_decoded_prob), cmap="magma")
p1 = axs[0].plot(smth_decoded_position, "--", color="limegreen")
p2 = axs[0].plot(position.restrict(ex_run_ep), color="limegreen")
axs[0].set_ylabel("Position (cm)")
axs[0].legend([p1[0],p2[0]],["decoded position","true position"])
fig.colorbar(c, label = "predicted probability")

##-- Also plot the raw and filtered LFP aligned to the decoding --##
axs[1].plot(lfp.restrict(ex_run_ep), label="raw")
axs[1].plot(theta_band.restrict(ex_run_ep), label="filtered")
axs[1].set_ylabel("LFP (a.u.)")
axs[1].legend()

fig.supxlabel("Time (s)");
```

## Fitting a GLM

```{code-cell} ipython3
position_up = data["position"].interpolate(theta_phase).dropna()

speed = []
for s, e in position_up.time_support.values: # Time support contains the epochs
    pos_ep = position_up.get(s, e)
    speed_ep = np.abs(np.diff(pos_ep)) # Absolute difference of two consecutive points
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") # Adding one point at the end to match the size of the position array
    speed_ep = speed_ep * sample_rate # Converting to cm/s
    speed.append(speed_ep)

speed = nap.Tsd(t=position_up.t, d=np.hstack(speed), time_support=position_up.time_support)
```

```{code-cell} ipython3
predictors = np.stack((position_up, speed, theta_phase.restrict(position_up.time_support)))
X = nap.TsdFrame(t=position_up.t, d=predictors.T, time_support=position_up.time_support)

unit = 117
bin_size = 0.001
counts = spikes[unit].count(bin_size, ep=X.time_support)
X = X.bin_average(bin_size)

model = nmo.glm.GLM(solver_name="LBFGS")
model.fit(X, counts)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
counts.shape
```
