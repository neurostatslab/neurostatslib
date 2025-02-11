---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.6
kernelspec:
  display_name: test
  language: python
  name: python3
---

# Introduction
Paper: https://www.sciencedirect.com/science/article/pii/S0092867423014459

- **Subject**: mice, transgenic
- **Number**: 28

### Recording details
- **Recording type**: acute electrophysiology via Neuropixels 1.0 probes
    - this means that probes are inserted at the beginning of every session and retrieved at the end. recordings will not be taken from the exact same location twice. mice can usually be recorded from for up to 5 sessions using this technique, after which the insertion windows are no longer viable
    - ALSO: intermittent optogenetic inactivation of ALM
- **Probe density**: 2-5 simultaneous probes per recording
    - 2 probes - 2 sessions
    - 3 probes - 53 sessions
    - 4 probes - 98 sessions
    - 5 probes - 20 sessions
- **Recording targets**: anterior lateral motor cortex (ALM) circuit. individual probes target the following groups: 
    1. ALM and underlying cortical regions (e.g. orbito frontal cortex)
    2. striatum
    3. higher-order thalamus
    4. midbrain (superior colliculus (SCm), midbrain reticular nucleus (MRN), and substantia nigra pars reticulata (SNr))
    5. medulla and overlying cerebellum
    6. other
- **Spike-sorting**: Kilosort2 (probably MATLAB)
- **Unit count**: 70,000 total single units, localized using hisological information and electrophysiological targets. Median of 393 simultaneously recorded units per session.
    
### Behavior details
- **Headfixed task**: Memory-guided movement task (i.e. auditory delayed response task)
    - instruction stimuli: one of two pure tones (3 kHz or 12 kHz) played three times, 150 ms pulses and 100 ms inter-tone-interval, 650 ms total
    - delay epoch: 1.2s
    - can't lick until auditory 'Go' cue, 6 kHz carrier frequency with 360 Hz modulation, 0.1 s duration, where early licking triggered replay of delay epoch
    - response epoch: 1.5 s, correct lick triggered small water reward
    - incorrect licks triggered a 1-3 s timeout
    - trial ends after mouse stops licking for 1.5 s, followed by a 250 ms inter-trial-interval
    - early lick and no-response trials excluded from analysis
- **Video tracking**: 300Hz recording from two cameras to capture animal movements
    - offline tracking of tongue, jaw, and nose using DeepLabCut

```{code-cell} ipython3
import pynacollada as nac
import pynapple as nap
```

# The data

## Acquire / Download

DANDI archive: https://dandiarchive.org/dandiset/000363/0.230822.0128

**size**: 53.6 GB

download to lab group folder? 

instructions outside of notebook?

## File structure
- 28 folders - one for each mouse
    - `.nwb` files: one for each session with naming scheme `sub-{number}_ses-{YYYYMMDD}T{HHMMSS}_behavior+ecephys[+ogen]` (most files have `+ogen` on the end, signalizing optogenetics done that session)
    
NWB file:

```{code-cell} ipython3
nac.config["data_dir"] = "../data"
data = nac.load_data("mesoscale_activity")
data
```

# Pynapple

## Importing the data

+++

### Load using pynapple (allows for lazy loading)

can load directly using `nap.load_file()`, which does a good job

```{code-cell} ipython3
data = nap.load_file(os.path.join(fpath,fname))
print(data)
```

```{code-cell} ipython3
data['trials']
```

```{code-cell} ipython3
data['units']
```

### Create from loaded .nwb file
Allows us to manipulate what types of objects are created, things that can't be inferred necessary from parsing the file

Grab trials as a dataframe and transform into interval set. This will be the same as what's loaded above

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
---
trials = nwbfile.trials.to_dataframe()
trials = trials.rename(columns={'start_time':'start','stop_time':'end'})
trials = nap.IntervalSet(trials)
```

```{code-cell} ipython3
trials
```

variables saved in BehaviorEvents often have start and stop times, where the "data" value is meaningless. It would make more sense to combine them and create an interval set for each. other variables are only timestamps, with similarly meaningless "data" values associated with them. they would be better as Ts objects

grab other time stamps from BehavioralEvents and put into a dictionary. Do some manipulation to concatenate start and stop times into a dataframe

```{code-cell} ipython3
events = {}
beh = nwbfile.acquisition['BehavioralEvents'].time_series
for key in beh:
    if 'start' in key:
        key2 = key.replace('start_','')
        if key2 not in events.keys():
            events[key2] = pd.DataFrame(columns=['start','end'])
        events[key2]['start'] = beh[key].timestamps[:]
    elif 'stop' in key:    
        key2 = key.replace('stop_','')
        if key2 not in events.keys():
            events[key2] = pd.DataFrame(columns=['start','end'])
        events[key2]['end'] = beh[key].timestamps[:]
    else:
        events[key] = beh[key].timestamps[:]
```

turn into pynapple objects

```{code-cell} ipython3
for key in events:
    if isinstance(events[key],pd.DataFrame):
        events[key] = nap.IntervalSet(events[key])
    else:
        events[key] = nap.Ts(events[key])
```

### Spiking data as TsGroup

+++

grab spiking data and import into pynapple TsGroup, preserving metadata. this should also match the pynapple loaded objects

```{code-cell} ipython3
---
jupyter:
  outputs_hidden: true
---
units = nwbfile.units.to_dataframe()
spike_times = df["spike_times"]
metadata = df.drop(columns="spike_times")
units = nap.TsGroup(spike_times)
units.set_info(metadata)
```

```{code-cell} ipython3
units
```

## Basic time series analysis
### Binning

### Tuning curves

### Bayesian decoding

+++

# NeMoS

```{code-cell} ipython3
import nemos as nmo
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
