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
    
In this tutorial we will learn how to use more advanced applications of pynapple: signal processing and decoding. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.

```{code-cell} ipython3
:tags: [render-all]

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

# necessary for animation
import nemos as nmo
plt.style.use(nmo.styles.plot_style)
```

***

## Fetching the data
    
The data set we'll be looking at is from the manuscript [Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences](https://www.science.org/doi/10.1126/science.aad1935). In this study, the authors collected electrophisiology data in rats across multiple sites in layer CA1 of hippocampus to extract the LFP alongside spiking activity of many simultaneous pyramidal units. In each recording session, data were collected while the rats explored a novel environment (a linear or circular track), as well as during sleep before and after exploration. In our following analyses, we'll focus on the exploration period of a single rat and recording session.

The full dataset for this study can be accessed on [DANDI](https://dandiarchive.org/dandiset/000044/0.210812.1516). Since the file size of a recording session can be large from the LFP saved for each recorded channel, we'll use a smaller file that contains the spiking activity and the LFP from a single, representative channel, which is hosted on [OSF](https://osf.io/2dfvp). This smaller file, like the original data, is saved as an [NWB](https://www.nwb.org) file.

```{code-cell} ipython3
# load pynacollada data set
data = nac.load_data("place_cells")
print(data)
```

This function will give us the file path to where the data is stored. We can then use the pynapple function `load_file` to load in the data, which is able to handle the NWB file type.

This returns a dictionary of pynapple objects that have been extracted from the NWB file. Let's explore each of these objects.

:::{admonition} Note
:class: note render-all
We will ignore the object `theta_phase` because we will be computing this ourselves later on in the exercise.
:::


### units

<div class="render-all">  
    
The `units` field is a `TsGroup`: a collection of `Ts` objects containing the spike times of each unit, where the "Index" is the unit number or key. Each unit has the following metadata:
- **rate**: computed by pynapple, is the average firing rate of the neuron across all recorded time points.
- **location**, **shank**, and **cell_type**: variables saved and imported from the original data set.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["units"]
```

<div class="render-all">  

We can access the spike times of a single unit by indexing the `TsGroup` by its unit number. For example, to access the spike times of unit 1:

</div>

```{code-cell} ipython3
:tags: [render-all]

data["units"][1]
```

### rem, nrem, and forward_ep

<div class="render-all">  

The next three objects; `rem`, `nrem`, and `forward_ep`; are all IntervalSets containing time windows of REM sleep, nREM sleep, and forward runs down the linear maze, respectively. 

</div>

```{code-cell} ipython3
:tags: [render-all]

data["rem"]
```

```{code-cell} ipython3
:tags: [render-all]

data["nrem"]
```

```{code-cell} ipython3
:tags: [render-all]

data["forward_ep"]
```

All intervals in `forward_ep` occur in the middle of the session, while `rem` and `nrem` both contain sleep epochs that occur before and after exploration. 

<div class="render-presenter">
- sleep epochs are intertwined, forward epoch in middle
</div>

<div class="render-all"> 
    
The following plot demonstrates how each of these labelled epochs are organized across the session.

</div>

```{code-cell} ipython3
:tags: [render-all]

t_start = data["nrem"].start[0]
fig,ax = plt.subplots(figsize=(10,2), constrained_layout=True)
sp1 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="blue", alpha=0.1) for iset in data["rem"]];
sp2 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="green", alpha=0.1) for iset in data["nrem"]];
sp3 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="red", alpha=0.1) for iset in data["forward_ep"]];
ax.set(xlabel="Time within session (minutes)", title="Labelled time intervals across session", yticks=[])
ax.legend([sp1[0],sp2[0],sp3[0]], ["REM sleep","nREM sleep","forward runs"]);
```

### eeg

<div class="render-all">  

The `eeg` object is a `TsdFrame` containing an LFP voltage trace for a single representative channel in CA1.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["eeg"]
```

<div class="render-all">  

Despite having a single column, this `TsdFrame` is still a 2D object. We can represent this as a 1D `Tsd` by indexing into the first column.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["eeg"][:,0]
```

### position

<div class="render-all">  

The final object, `position`, is a `Tsd` containing the linearized position of the animal, in centimeters, recorded during the exploration window.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["position"]
```

<div class="render-all">  

Positions that are not defined, i.e. when the animal is at rest, are filled with `NaN`.

This object additionally contains a `time_support` attribute, which gives the time interval during which positions are recorded (including points recorded as `NaN`).

</div>

```{code-cell} ipython3
:tags: [render-all]

data["position"].time_support
```

<div class="render-all">  

Let's visualize the first 300 seconds of position data and overlay `forward_ep` intervals.

</div>

```{code-cell} ipython3
pos_start = data["position"].time_support.start[0]
fig, ax = plt.subplots(figsize=(10,3))
l1 = ax.plot(data["position"])
l2 = [ax.axvspan(iset.start[0], iset.end[0], color="red", alpha=0.1) for iset in data["forward_ep"]];
ax.set(xlim=[pos_start,pos_start+300], ylabel="Position (cm)", xlabel="Time (s)", title="Tracked position along linear maze")
ax.legend([l1[0], l2[0]], ["animal position", "forward run epochs"])
```

This plot confirms that positions are only recorded while the animal is moving along the track. Additionally, it is clear that the intervals in `forward_ep` capture only perios when the animal's position is increasing, during forward runs.

<div class="render-presenter">
- position only while moving
- `forward_ep` only captures forward runs
</div>

## Restricting the data

<div class="render-all">
    
For the following exercises, we'll only focus on periods when the animal is awake. We'll start by pulling out `forward_ep` from the data.

</div>

```{code-cell} ipython3
awake_ep = data["position"].time_support
```

For the second, we know that the animal is likely at rest when there is no recorded position (i.e. the position is NaN). We can create this `IntervalSet`, then, using the following steps.

1. Drop `NaN` values from the position to grab only points where position is defined.

</div>

<div class="render-user">
```{code-cell} ipython3
# drop nan values
pos_good = 
```
</div>

```{code-cell} ipython3
# drop nan values
pos_good = data["position"].dropna()
pos_good
```

<div class="render-all">

2. Extract time intervals from `pos_good` using the `find_support` method
   - The first input argument, `min_gap`, sets the minumum separation between adjacent intervals in order to be split
   - Here, use `min_gap` of 1 s

</div>

<div class="render-user">
```{code-cell} ipython3
# extract time support
position_ep = 
```
</div>

```{code-cell} ipython3
# extract time support
position_ep = pos_good.find_support(1)
position_ep
```

<div class="render-all">

3. Define resting epochs as the set difference between `awake_ep` and `position_ep`, using the `set_diff` method.
   - `set_diff` should be applied to `awake_ep`, not the other way around, such that intervals in `position_ep` are subtracted out of `awake_ep`

</div>

<div class="render-user">
```{code-cell} ipython3
rest_ep = 
```
</div>

```{code-cell} ipython3
rest_ep = awake_ep.set_diff(position_ep)
rest_ep
```

:::{admonition} Note
:class: note render-all

Performing `set_diff` between `awake_ep` and `forward_ep` will *not* give us purely resting epochs, since these intervals will also include times when the animal is moving *backwards* across the linear track.

:::

<div class="render-all">
    
Now, when extracting the LFP, spikes, and position, we can use `restrict()` with `awake_ep` to restrict the data to our region of interest.

</div>

```{code-cell} ipython3
:tags: [render-all]

lfp_run = data["eeg"][:,0].restrict(awake_ep)
spikes = data["units"].restrict(awake_ep)
position = data["position"].restrict(awake_ep)
```

<div class="render-all">
    
For visualization, we'll look at a single run down the linear track. For a good example, we'll start by looking at run 10 (python index 9). Furthermore, we'll add two seconds on the end of the run to additionally visualize a period of rest following the run.
    
</div>

```{code-cell} ipython3
:tags: [render-all]

ex_run_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end+2)
ex_run_ep
```

***

## Plotting the LFP and animal position

<div class="render-all">

To get a sense of what the LFP looks like while the animal runs down the linear track, we can plot each variable, `lfp_run` and `position`, side-by-side. 

We'll want to further restrict each variable to our run of interest, `ex_run_ep`.

</div>

<div class="render-user">
```{code-cell} ipython3
ex_lfp_run = 
ex_position = 
```
</div>

```{code-cell} ipython3
ex_lfp_run = lfp_run.restrict(ex_run_ep)
ex_position = position.restrict(ex_run_ep)
```

<div class="render-all">

Let's plot the example LFP trace and anmimal position. Plotting `Tsd` objects will automatically put time on the x-axis.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 4), sharex=True)

# plot LFP
axs[0].plot(ex_lfp_run)
axs[0].set_title("Local Field Potential on Linear Track")
axs[0].set_ylabel("LFP (a.u.)")

# plot animal's position
axs[1].plot(ex_position)
axs[1].set_title("Animal Position on Linear Track")
axs[1].set_ylabel("Position (cm)") # LOOK UP UNITS
axs[1].set_xlabel("Time (s)");
```

As we would expect, there is a strong theta oscillation dominating the LFP while the animal runs down the track. This oscillation is weaker after the run is complete.

<div class="render-presenter">
- theta while running, weaker after run is complete
</div>

***

## Getting the Wavelet Decomposition

<div class="render-all">

To illustrate this further, we'll perform a wavelet decomposition on the LFP trace during this run. We can do this in pynapple using the function `nap.compute_wavelet_transform`. This function takes the following inputs (in order):
- `sig`: the input signal; a `Tsd`, a `TsdFrame`, or a `TsdTensor`
- `freqs`: a 1D array of frequency values to decompose

We will also supply the following optional arguments:
- `fs`: the sampling rate of `sig`

</div>

A [continuous wavelet transform](https://en.wikipedia.org/wiki/Continuous_wavelet_transform) decomposes a signal into a set of [wavelets](https://en.wikipedia.org/wiki/Wavelet), in this case [Morlet wavelets](https://en.wikipedia.org/wiki/Morlet_wavelet), that span both frequency and time. You can think of the wavelet transform as a cross-correlation between the signal and each wavelet, giving the similarity between the signal and various frequency components at each time point of the signal. Similar to a Fourier transform, this gives us an estimate of what frequencies are dominating a signal. Unlike the Fourier tranform, however, the wavelet transform gives us this estimate as a function of time.

We must define the frequency set that we'd like to use for our decomposition. We can do this with the numpy function `np.geomspace`, which returns numbers evenly spaced on a log scale. We pass the lower frequency, the upper frequency, and number of samples as positional arguments.

<div class="render-presenter">
Wavelet transform:
- continuous wavelet transform decomposes signal into Morlet wavelets spanning frequency and time
- like cross-correlation between signal and wavelets, giving similarity between the signal and a short oscillating wavelet at different points in time
- give estimate fo what frequencies are dominating a signal
- unlike FFT, gives estimate as a function of time
</div>

<div class="render-user render-presenter">
1. Define 100 log-spaced samples between 5 and 200 Hz using `np.geomspace`
</div>

<div class="render-user">
```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = 
```
</div>

```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = np.geomspace(5, 200, 100)
```

We can now compute the wavelet transform on our LFP data during the example run using `nap.compute_wavelet_trasform` by passing both `ex_lfp_run` and `freqs`. We'll also pass the optional argument `fs`, which is known to be 1250Hz from the study methods.

<div class="render-user render-presenter">
2. Compute the wavelet transform, supplying the known sampling rate of 1250 Hz.
</div>

<div class="render-user">  
```{code-cell} ipython3
sample_rate = 1250
cwt_run =
```
</div>

```{code-cell} ipython3
sample_rate = 1250
cwt_run = nap.compute_wavelet_transform(ex_lfp_run, freqs, fs=sample_rate)
```

<div class="render-all">

If `fs` is not provided, it can be inferred from the time series `rate` attribute.

</div>

```{code-cell} ipython3
:tags: [render-all]

print(ex_lfp_run.rate)
```

The inferred rate is close to the true sampling rate, but it can introduce a small floating-point error. Therefore, it is better to supply the true sampling rate when it is known.

<div class="render-presenter">
- note floating point error
</div>

<div class="render-all">

We can visualize the results by plotting a heat map of the calculated wavelet scalogram.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, height_ratios=[1.0, 0.3], sharex=True)
fig.suptitle("Wavelet Decomposition")

amp = np.abs(cwt_run.values)
cax = axs[0].pcolormesh(cwt_run.t, freqs, amp.T)
axs[0].set(ylabel="Frequency (Hz)", yscale='log', yticks=freqs[::10], yticklabels=np.rint(freqs[::10]));
axs[0].minorticks_off()
fig.colorbar(cax,label="Amplitude")

p1 = axs[1].plot(ex_lfp_run)
axs[1].set(ylabel="LFP (a.u.)", xlabel="Time(s)")
axs[1].margins(0)
ax = axs[1].twinx()
p2 = ax.plot(ex_position, color="orange")
ax.set_ylabel("Position (cm)")
ax.legend([p1[0], p2[0]],["raw LFP","animal position"])
```

As we would expect, there is a strong presence of theta in the 6-12Hz frequency band while the animal runs down the track, which dampens during rest.

<div class="render-presenter">
- strong amplitude in 6-12Hz range while animal is running, dampens after
</div>

***
## Bonus: Additional signal processing methods
<div class="render-all">
    
- `nap.compute_fft`
  
</div>

```{code-cell} ipython3
:tags: [render-all]

fft_amp = np.abs(nap.compute_fft(lfp_run, fs=sample_rate, norm=True))
fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
ax.plot(fft_amp[(fft_amp.index >= 1) & (fft_amp.index <= 100)])
ax.axvspan(6, 12, color="red", alpha=0.1, label = "theta band")
ax.set(xlabel="Frequency (Hz)", ylabel="Normalized Amplitude (a.u.)", title="FFT amplitude during the awake epoch")
fig.legend(loc="center")
```

<div class="render-all">
    
- `nap.compute_power_spectral_density`
  
</div>

```{code-cell} ipython3
:tags: [render-all]

power = nap.compute_power_spectral_density(lfp_run, fs=sample_rate)
fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
ax.plot(power[(power.index >= 1) & (power.index <= 100)])
ax.axvspan(6, 12, color="red", alpha=0.1, label = "theta band")
ax.set(xlabel="Frequency (Hz)", ylabel="Power/Frequency (a.u./Hz)", title="Periodogram during the awake epoch")
fig.legend(loc="center")
```

***
## Filtering for theta

<div class="render-all">

For the remaining exercises, we'll reduce our example epoch to the portion when the animal is running down the linear track.

</div>

```{code-cell} ipython3
:tags: [render-all]

ex_run_ep = forward_ep[9]
ex_lfp_run = lfp_run.restrict(ex_run_ep)
ex_position = position.restrict(ex_run_ep)
```

We can extract the theta oscillation by applying a bandpass filter on the raw LFP. To do this, we use the pynapple function `nap.apply_bandpass_filter`, which takes the time series as the first argument and the frequency cutoffs as the second argument. Similarly to `nap.compute_wavelet_transorm`, we can optinally pass the sampling frequency keyword argument `fs`.

Conveniently, this function will recognize and handle splits in the epoched data (i.e. applying the filtering separately to discontinuous epochs), so we don't have to worry about passing signals that have been split in time.

<div class="render-user render-presenter">
We can filter our signal for theta by using `nap.apply_bandpass_filter`, which requires following arguments:
- `data`: the signal to be filtered; a `Tsd`, `TsdFrame`, or `TsdTensor`
- `cutoff`: tuple containing the frequency cutoffs, (lower frequency, upper frequency)

Same as before, we'll pass the optional argument:
- `fs`: the sampling rate of `data` in Hz

Using this function, filter `lfp_run` within a 6-12 Hz range.
</div>

<div class="render-presenter">
- note handling of disconinuous data
</div>

<div class="render-user">   
```{code-cell} ipython3
theta_band = 
```
</div>

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp_run, (6.0, 12.0), fs=sample_rate)
```

<div class="render-all">

We can visualize the output by plotting the filtered signal with the original signal.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure(constrained_layout=True, figsize=(10, 3))
plt.plot(ex_lfp_run, label="raw")
plt.plot(theta_band.restrict(ex_run_ep), label="filtered")
plt.xlabel("Time (s)")
plt.ylabel("LFP (a.u.)")
plt.title("Bandpass filter for theta oscillations (6-12 Hz)")
plt.legend();
```

***
## Computing theta phase

<div class="render-all">

In order to examine phase precession in place cells, we need to extract the phase of theta from the filtered signal. We can do this by taking the angle of the [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform).

The `signal` module of `scipy` includes a function to perform the Hilbert transform, after which we can use the numpy function `np.angle` to extract the angle.

</div>

<div class="render-user"> 
```{code-cell} ipython3
phase = 
```
</div>

```{code-cell} ipython3
phase = np.angle(signal.hilbert(theta_band)) # compute phase with hilbert transform
phase
```

<div class="render-all">

The output angle will be in the range $-\pi$ to $\pi$. Converting this to a $0$ to $2\pi$ range instead, by adding $2\pi$ to negative angles, will make later visualization more interpretable.

</div>

```{code-cell} ipython3
phase[phase < 0] += 2 * np.pi # wrap to [0,2pi]
```

<div class="render-all">

Finally, we need to turn this into a `Tsd` to make full use of pynapple's conveniences! Do this using the time index of `theta_band`. 

</div>

<div class="render-user">  
```{code-cell} ipython3
theta_phase = 
```
</div>

```{code-cell} ipython3
theta_phase = nap.Tsd(t=theta_band.t, d=phase)
```

<div class="render-all">

Let's plot the phase on top of the filtered LFP signal.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2,1,figsize=(10,4), constrained_layout=True) #, sharex=True, height_ratios=[2,1])

ax = axs[0]
ax.plot(ex_lfp_run)

ax = axs[1]
p1 = ax.plot(theta_phase.restrict(ex_run_ep), color='r')
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_ep))
ax.set_ylabel("LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```

<div class="render-all">

Let's zoom in on a few cycles to get a better look.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10,4), constrained_layout=True) #, sharex=True, height_ratios=[2,1])

ex_run_shorter = nap.IntervalSet(ex_run_ep.start[0], ex_run_ep.start[0]+0.25)

p1 = ax.plot(theta_phase.restrict(ex_run_shorter), color='r')
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_shorter))
ax.set_ylabel("LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```

We can see that cycle "resets" (i.e. goes from $2\pi$ to $0$) at peaks of the theta oscillation.

<div class="render-presenter">
- note cycle reset
</div>

***

## Computing 1D tuning curves: place fields

<div class="render-all">

In order to identify phase precession in single units, we need to know their place selectivity. We can find place firing preferences of each unit by using the function `nap.compute_1d_tuning_curves`. This function has the following required inputs:
- `group`: a `TsGroup` of units for which tuning curves are computed
- `feature`: a `Tsd` or single-column `TsdFrame` of the feature over which tuning curves are computed (e.g. position)
- `nb_bins`: the number of bins in which to split the feature values for the tuning curve

First, we'll filter for units that fire at least 1 Hz and at most 10 Hz when the animal is running forward along the linear track. This will select for units that are active during our window of interest and eliminate putative interneurons (i.e. fast-firing inhibitory neurons that don't usually have place selectivity). 

</div>

<div class="render-user render-presenter">
1. Restrict `spikes` to `forward_ep`
</div>


<div class="render-user">
```{code-cell} ipython3
forward_spikes =
```
</div>

<div class="render-presenter">
```{code-cell} ipython3
forward_spikes = spikes.restrict(forward_ep)
```
</div>

<div class="render-user render-presenter">
2. Select for units whose rate is at least 1 Hz and at most 10 Hz
</div>

<div class="render-user">
```{code-cell} ipython3
good_spikes = 
```
</div>

<div class="render-presenter">
```{code-cell} ipython3
good_units = (forward_spikes.rate >= 1) & (forward_spikes.rate <= 10)
good_spikes = forward_spikes[good_units]
```
</div>

```{code-cell} ipython3
good_spikes = spikes[(spikes.restrict(forward_ep).rate >= 1) & (spikes.restrict(forward_ep).rate <= 10)]
```

<div class="render-all">

Using these units and the position data, we can compute their place fields using `nap.compute_1d_tuning_curves`. This function will return a `pandas.DataFrame`, where the index is the corresponding feature value, and the column is the unit label. Let's compute this for 50 position bins.

</div>

:::{admonition} Tip
:class: tip render-all

The reason `nap.compute_1d_tuning_curves` returns a `pandas.DataFrame` and not a Pynapple object is because the index corresponds to the *feature*, where all Pynapple objects assume the index is *time*.
:::

<div class="render-user">
```{code-cell} ipython3
place_fields = 
```
</div>

```{code-cell} ipython3
place_fields = nap.compute_1d_tuning_curves(good_spikes, position, 50)
```

<div class="render-all">

We can use a subplot array to visualize the place fields of many units simultaneously. Let's do this for the first 50 units.

</div>

```{code-cell} ipython3
:tags: [render-all]

from scipy.ndimage import gaussian_filter1d

# smooth the place fields so they look nice
place_fields[:] = gaussian_filter1d(place_fields.values, 1, axis=0)

fig, axs = plt.subplots(10, 5, figsize=(12, 15), sharex=True, constrained_layout=True)
for i, (f, fields) in enumerate(place_fields.iloc[:,:50].items()):
    idx = np.unravel_index(i, axs.shape)
    axs[idx].plot(fields)
    axs[idx].set_title(f)

fig.supylabel("Firing rate (Hz)")
fig.supxlabel("Position (cm)")
```

We can see spatial selectivity in each of the units; across the population, we have firing fields tiling the entire linear track. 

<div class="render-presenter">
- note representations cover entire track
</div>

***
## Visualizing phase precession within a single unit

As an initial visualization of phase precession, we'll look at a single traversal of the linear track. First, let's look at how the timing of an example unit's spikes lines up with the LFP and theta. To plot the spike times on the same axis as the LFP, we'll use the pynapple object's method `value_from` to align the spike times with the theta amplitude. For our spiking data, this will find the amplitude closest in time to each spike. Let's start by applying `value_from` on unit 177, who's place field is cenetered on the linear track, using `theta_band` to align the amplityde of the filtered LFP.

<div class="render-user render-presenter">
First, let's look at how an example unit fires with respect to the filtered LFP. Using the pynapple object method `value_from`, we can find the value of `theta_band` corresponding to each spike time. Let's do this for unit 177, who's place field is cenetered on the linear track.
</div>

<div class="render-user">  
```{code-cell} ipython3
unit = 177
spike_theta = 
```
</div>

```{code-cell} ipython3
unit = 177
spike_theta = spikes[unit].value_from(theta_band)
```

<div class="render-all">

Let's plot `spike_theta` on top of the LFP and filtered theta, as well as visualize the animal's position along the track.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, sharex=True)
axs[0].plot(ex_lfp_run, alpha=0.5, label="raw LFP")
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)")
axs[0].legend()

axs[1].plot(ex_position, '--', color="green", label="animal position")
axs[1].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)")
axs[1].legend()
```

As expected, unit 177 will preferentially spike (orange dots) as the animal runs through the middle of the track (thick green). You may be able to notice that the spikes are firing at specific points in each theta cycle, and these points are systematically changing over time. At first, spikes fire along the rising edge of a cycle, afterwards firing earlier towards the trough, and finally firing earliest along the falling edge.

We can exemplify this pattern by plotting the spike times aligned to the phase of theta. We'll want the corresponding phase of theta at which the unit fires as the animal is running down the track, which we can again compute using the method `value_from`. 

<div class="render-user render-presenter">
As the animal runs through unit 177's place field (thick green), the unit spikes (orange dots) at specific points along the theta cycle dependent on position: starting at the rising edge, moving towards the trough, and ending at the falling edge.

We can exemplify this pattern by plotting the spike times aligned to the phase of theta. Let's compute the phase at which each spike occurs by using `value_from` with `theta_phase`. 
</div>

<div class="render-user">  
```{code-cell} ipython3
spike_phase = 
```
</div>

```{code-cell} ipython3
spike_phase = spikes[unit].value_from(theta_phase)
```

<div class="render-all">

To visualize the results, we'll recreate the plot above, but instead with the theta phase.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(3, 1, figsize=(10,6), constrained_layout=True, sharex=True)
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)", title="Spike times relative to filtered theta")
axs[0].legend()

axs[1].plot(theta_phase.restrict(ex_run_ep), color="slateblue", label="theta phase")
axs[1].plot(spike_phase.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[1].set(ylabel="Phase (rad)", title="Spike times relative to theta phase")
axs[1].legend()

axs[2].plot(ex_position, '--', color="green", label="animal position")
axs[2].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[2].set(ylabel="Position (cm)", xlabel="Time (s)", title="Animal position")
axs[2].legend()
```

You should now see a negative trend in the spike phase as the animal moves along the track. This phemomena is what is called phase precession: as an animal runs through the place field of a single unit, that unit will spike at *late* phases of theta (higher radians) in *earlier* positions in the field, and fire at *early* phases of theta (lower radians) in *late* positions in the field.

We can observe this phenomena on average across all runs by relating the spike phase to the spike position. Similar to before, we'll use the pynapple object method `value_from` to additionally find the animal position closest in time to each spike.

<div class="render-user render-presenter">
We now see a negative trend in the spike phase as the animal moves through unit 177's place field, indicative of this unit *phase precessing*. 

We can observe this phenomena on average across the session by relating the spike phase to the spike position. Try computing the spike position from what we've learned so far.
</div>

<div class="render-user">
```{code-cell} ipython3
spike_position = 
```
</div>

```{code-cell} ipython3
spike_position = spikes[unit].value_from(position)
```

<div class="render-all">

Now we can plot the spike phase against the spike position in a scatter plot.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.subplots(figsize=(5,3))
plt.plot(spike_position, spike_phase, 'o')
plt.ylabel("Phase (rad)")
plt.xlabel("Position (cm)")
```

Similar to what we saw in a single run, there is a negative relationship between theta phase and field position, characteristic of phase precession.

<div class="render-presenter">
- note negative relationship
</div>

***
## Computing 2D tuning curves: position vs. phase
<div class="render-all">

The scatter plot above can be similarly be represented as a 2D tuning curve over position and phase. We can compute this using the function `nap.compute_2d_tuning_curves`. This function requires the same inputs as `nap.compute_1d_tuning_curves`, except now the second input, `features`, must be a 2-column `TsdFrame` containing the two target features.

To use this function, we'll need to combine `position` and `theta_phase` into a `TsdFrame`. To do this, both variables must have the same length. We can achieve this by upsampling `position` to the length of `theta_phase` using the pynapple object method `interpolate`. This method will linearly interpolate new position samples between existing position samples at timestamps given by another pynapple object, in our case by `theta_phase`.

</div>

<div class="render-user"> 
```{code-cell} ipython3
upsampled_pos = 
```
</div>

```{code-cell} ipython3
upsampled_pos = position.interpolate(theta_phase)
```

<div class="render-all">

Let's visualize the results of the interpolation.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2,1,constrained_layout=True,sharex=True,figsize=(10,4))
axs[0].plot(position.restrict(ex_run_ep),'.')
axs[0].set(ylabel="Position (cm)", title="Original position points")
axs[1].plot(upsampled_pos.restrict(ex_run_ep),'.')
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)", title="Upsampled position points")
```

<div class="render-all">

We can now stack `upsampled_pos` and `theta_phase` into a single array.

</div>

<div class="render-user">  
```{code-cell} ipython3
feats = 
```
</div>

```{code-cell} ipython3
feats = np.stack((upsampled_pos.values, theta_phase.values))
feats.shape
```

<div class="render-all">

Using `feats`, we can define a `TsdFrame` using the time index from `theta_phase` and the time support from `upsampled_pos`. Note that `feats` has the wrong shape; we want time in the first dimension, so we'll need to pass its transpose.

</div>

<div class="render-user">
```{code-cell} ipython3
features =
```
</div>

```{code-cell} ipython3
features = nap.TsdFrame(
    t=theta_phase.t,
    d=np.transpose(feats),
    time_support=upsampled_pos.time_support,
    columns=["position", "theta"],
)
```

<div class="render-all">

Now we have what we need to compute 2D tuning curves. Let's apply `nap.compute_2d_tuning_curves` on our reduced group of units, `good_spikes`, using 20 bins for each feature. 

This function will return two outputs:
1. A dictionary of the 2D tuning curves, where dictionary keys correspond to the unit label
2. A list with length 2 containing the feature bin centers
   
</div>

<div class="render-user">
```{code-cell} ipython3
tuning_curves, [pos_x, phase_y] =
```

</div>

```{code-cell} ipython3
tuning_curves, [pos_x, phase_y] = nap.compute_2d_tuning_curves(good_spikes, features, 20)
```

<div class="render-all">

We can plot the first 50 2D tuning curves and visualize how many of these units are phase precessing.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(10, 5, figsize=(10, 15), sharex=True, constrained_layout=True)
for i, f in enumerate(list(tuning_curves.keys())[:50]):
    idx = np.unravel_index(i, axs.shape)
    axs[idx].pcolormesh(pos_x, phase_y, tuning_curves[f])
    axs[idx].set_title(f)

fig.supylabel("Phase (rad)")
fig.supxlabel("Position (cm)");
```

Many of the units display a negative relationship between position and phase, characteristic of phase precession.

<div class="render-presenter">
- point out some of the precessing units
</div>

***
## Decoding position from spiking activity

<div class="render-all">

Next we'll do a popular analysis in the rat hippocampal sphere: Bayesian decoding. This analysis is an elegent application of Bayes' rule in predicting the animal's location (or other behavioral variables) from neural activity at some point in time. 

</div>

### Background

<div class="render-user render-presenter">
For a more in-depth background on Bayesian decoding, see the full version of this notebook online.
</div>

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

If this method looks daunting, we have some good news: pynapple has it implemented already in the function `nap.decode_1d` for decoding a single dimension (or `nap.decode_2d` for two dimensions). All we'll need are the spikes, the tuning curves, and the width of the time window $\tau$.

### ASIDE: Cross-validation

Generally this method is cross-validated, which means you train the model on one set of data and test the model on a different, held-out data set. For Bayesian decoding, the "model" refers to the model *likelihood*, which is computed from the tuning curves. 

We want to decode the example run we've been using throughout this exercise; therefore, our training set should omit this run before computing the tuning curves. We can do this by using the IntervalSet method `set_diff`, to take out the example run epoch from all run epochs. Next, we'll restrict our data to these training epochs and re-compute the place fields using `nap.compute_1d_tuning_curves`. We'll also apply a Gaussian smoothing filter to the place fields, which will smooth our decoding results down the line.

:::{admonition} Important
:class: important render-user render-presenter

Generally this method is cross-validated, which means you train the model on one set of data and test the model on a different, held-out data set. For Bayesian decoding, the "model" refers to the model *likelihood*, which is computed from the tuning curves. Run the code below if you want to use a separate training set to compute the tuning curves.

:::

```{code-cell} ipython3
:tags: [render-all]

# hold out trial from place field computation
run_train = forward_ep.set_diff(ex_run_ep)
# get position of training set
position_train = position.restrict(run_train)
# compute place fields using training set
place_fields = nap.compute_1d_tuning_curves(spikes, position_train, nb_bins=50)
# smooth place fields
place_fields[:] = gaussian_filter1d(place_fields.values, 1, axis=0)
```

### Run 1D decoder

<div class="render-all">

With a single dimension in our tuning curves (position), we can apply Bayesian decoding using the function `nap.decode_1d`. This function requires the following inputs:
- `tuning_curves`: a `pandas.DataFrame`, computed by `nap.compute_1d_tuning_curves`, with the tuning curves relative to the feature being decoded
- `group`: a `TsGroup` of spike times, or a `TsdFrame` of spike counts, for each unit in `tuning_curves`.
- `ep`: an `IntervalSet` containing the epoch to be decoded
- `bin_size`: the time length, in seconds, of each decoded bin. If `group` is a `TsGroup` of spike times, this determines how the spikes are binned in time. If `group` is a `TsdFrame` of spike counts, this should be the bin size used for the counts.

This function will return two outputs:
- a `Tsd` containing the decoded feature at each decoded time point
- a `TsdFrame` containing the decoded probability of each feature value at each decoded time point, where the column names are the corresponding feature values

</div>

<div class="render-user render-presenter">
To increase decoder accuracy, we'll want to use the tuning curves of all the units in `spikes`. Recompute `place_fields` using `nap.compute_1d_tuning_curves` for all available units. (You can skip this if you're using the cross-validated `place_fields` from above.)
</div>

<div class="render-user">
```{code-cell} ipython3
place_fields =
```
</div>

<div class="render-presenter">
```{code-cell} ipython3
place_fields = nap.compute_1d_tuning_curves(spikes, position, nb_bins=50)
```
</div>

<div class="render-all">
    
Let's decode position during `ex_run_ep` using 40 ms time bins.

</div>

<div class="render-user">
```{code-cell} ipython3
decoded_position, decoded_prob = 
```
</div>

```{code-cell} ipython3
decoded_position, decoded_prob = nap.decode_1d(place_fields, spikes, ex_run_ep, 0.04)
```

<div class="render-all">

Let's plot decoded position with the animal's true position. We'll overlay them on a heat map of the decoded probability to visualize the confidence of the decoder.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(decoded_position.index,place_fields.index,np.transpose(decoded_prob))
ax.plot(decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

While the decoder generally follows the animal's true position, there is still a lot of error in the decoder, especially later in the run. In the next section, we'll show how to improve our decoder estimate.

<div class="render-presenter">
- note decoder error
</div>

### Smooth spike counts

<div class="render-all">

One way to improve our decoder is to supply smoothed spike counts to `nap.decode_1d`. We can smooth the spike counts by convolving them with a kernel of ones; this is equivalent to applying a moving sum to adjacent bins, where the length of the kernel is the number of adjacent bins being added together. You can think of this as counting spikes in a *sliding window* that shifts in shorter increments than the window's width, resulting in bins that overlap. This combines the accuracy of using a wider time bin with the temporal resolution of a shorter time bin.

</div>

For example, let's say we want a sliding window of $200 ms$ that shifts by $40 ms$. This is equivalent to summing together 5 adjacent $40 ms$ bins, or convolving spike counts in $40 ms$ bins with a length-5 array of ones ($[1, 1, 1, 1, 1]$). Let's visualize this convolution.

<div class="render-presenter">
- example: 200ms sliding window that shifts by 40ms
</div>

```{code-cell} ipython3
:tags: [render-all]

ex_counts = spikes[unit].restrict(ex_run_ep).count(0.04)
workshop_utils.animate_1d_convolution(ex_counts, np.ones(5), tsd_label="original counts", kernel_label="moving sum", conv_label="convolved counts")
```

The count at each time point is computed by convolving the kernel (yellow), centered at that time point, with the original spike counts (blue). For a length-5 kernel of ones, this amounts to summing the counts in the center bin with two bins before and two bins after (shaded green, top). The result is an array of counts smoothed out in time (green, bottom).

Let's compute the smoothed counts for all units. If we want a sliding window of $200 ms$ shifted by $40 ms$, we'll need to first count the spikes in $40 ms$ bins. We only need the counts for the epoch we want to decode, so we will use spikes restricted to `ep_run_ep`.

<div class="render-presenter">
- convolve kernel (yellow) centered at time point with original counts (blue)
- sum counts in center bin with two before and two after (shaded green)
- result is smoothed counts (bottom green)
</div>

<div class="render-user render-presenter">
Let's compute the smoothed counts for all units.

1. On spike times restricted to `ep_run_ep`, count spikes in $40 ms$ bins using the pynapple object method `count`.
</div>

<div class="render-user">
```{code-cell} ipython3
counts = 
```
</div>

```{code-cell} ipython3
counts = spikes.restrict(ex_run_ep).count(0.04)
```

Now, we need to sum each set of 5 adjacent bins to get our full window width of $200 ms$. We can compute this by using the pynapple object method `convolve` with the kernel `np.ones(5)`.

<div class="render-user render-presenter">
2. Convolve the counts with the kernel `np.ones(5)` using the pynapple object method `convolve`.
</div>

<div class="render-user">
```{code-cell} ipython3
smth_counts = 
```
</div>

```{code-cell} ipython3
smth_counts = counts.convolve(np.ones(5))
```

<div class="render-all">

Now we can use `nap.decode_1d` again with our smoothed counts in place of the raw spike times. Note that the bin size we'll want to provide is $200 ms$, since this is the true width of each bin.

</div>

<div class="render-user">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = 
```
</div>

```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ex_run_ep, bin_size=0.2)
```

<div class="render-all">

Let's plot the results.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(smth_decoded_position.index,place_fields.index,np.transpose(smth_decoded_prob))
ax.plot(smth_decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

This gives us a much closer approximation of the animal's true position.

<div class="render-presenter">
- improved approximation
</div>

### Decoding theta sequences

<div class="render-all">

Units phase precessing together creates fast, spatial sequences around the animal's true position. We can reveal this by decoding at an even shorter time scale, which will appear as smooth errors in the decoder.

</div>

Let's repeat what we did above, but now with a $50 ms$ sliding window that shifts by $10 ms$.

<div class="render-user render-presenter">
1. Get smoothed counts for a sliding window of $50 ms$ shifted by $10 ms$.
</div>

<div class="render-user">
```{code-cell} ipython3
smth_counts = 
```
</div>

<div class="render-presenter">
```{code-cell} ipython3
counts = spikes.restrict(ex_run_ep).count(0.01)
smth_counts = counts.convolve(np.ones(5))
```
</div>

<div class="render-user render-presenter">
2. Use `nap.decode_1d` to get the smoothed decoded position.
</div>

<div class="render-user">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = 
```
</div>

<div class="render-presenter">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ex_run_ep, bin_size=0.05)
```
</div>

```{code-cell} ipython3
counts = spikes.restrict(ex_run_ep).count(0.01)
smth_counts = counts.convolve(np.ones(5))
smth_decoded_position, smth_decoded_prob = nap.decode_1d(place_fields, smth_counts, ex_run_ep, bin_size=0.05)
```

<div class="render-all">
    
We'll make the same plot as before to visualize the results, but plot it alongside the raw and filtered LFP.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True, height_ratios=[3,1], sharex=True)
c = axs[0].pcolormesh(smth_decoded_prob.index, smth_decoded_prob.columns, np.transpose(smth_decoded_prob))
p1 = axs[0].plot(smth_decoded_position, "--", color="r")
p2 = axs[0].plot(ex_position, color="r")
axs[0].set_ylabel("Position (cm)")
axs[0].legend([p1[0],p2[0]],["decoded position","true position"])
fig.colorbar(c, label = "predicted probability")

axs[1].plot(ex_lfp_run)
axs[1].plot(theta_band.restrict(ex_run_ep))
axs[1].set_ylabel("LFP (a.u.)")

fig.supxlabel("Time (s)");
```

The estimated position oscillates with cycles of theta, where each "sweep" is referred to as a "theta sequence". Fully understanding the properties of theta sequences and their role in learning, memory, and planning is an active topic of research in Neuroscience!

<div class="render-presenter">
- position oscillates within cycles of theta 
- "sweep" is a "theta sequence"
- active topic of research
</div>

```{code-cell} ipython3

```
