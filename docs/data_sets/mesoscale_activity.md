---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---



# Installation


## Setting up your python virtual environment on the cluster

### 1. Connect to rusty
If you're on the FI network,
```bash
ssh rusty
```
If you're on the guest network or offsite, follow the instructions found here: https://wiki.flatironinstitute.org/SCC/RemoteConnect. In short, you need to first connect to the FI gateway, which requires Google Authenticator for your cluster account.

Connecting to rusty will land you on the **login node**. We can proceed with step 2 (creating our virtual environment) on this node, but this node should **not** be used for anything computationally intense.

### 2. Use modules and set up a virtual environment
The FI cluster has a bunch of useful software already installed as **modules**. I would recommend using modules in tandem with python's `venv` to quickly set up the virtual environment. 

First, we'll want to load in a python module. We can load in the default version of python using:
```bash
module load python
```
At the time of writing, this will load python 3.10.13. If you want to load a specific version of python, you can search available modules using `module spider`.
```bash
module spider python

```
This will return a list of python modules (and some other info):
```
----------------------------------------------------------------------------
  python:
----------------------------------------------------------------------------
     Versions:
        python/3.8.12
        python/3.8.15
        python/3.8.16
        python/3.9.12
        python/3.9.15
        python/3.9.16
        python/3.9.18
        python/3.10.4
        python/3.10.8
        python/3.10.10
        python/3.10.13
        python/3.11.2
        python/3.11.7

```
We can load in a specific version of python as:
```bash
module load python/3.11.7
```
Everything that follows assumes and has been tested with the default version (3.10.13), but ideally should also work with newer versions. 

> **Q: do we want to load in cuda for gpu support? need to check how gpus are handled with online jupyter scheduler**

+++

<div class="alert alert-block alert-info">
<b>Warning:</b> conda
</div>

dependencies:
- pynapple
- pynwb
- dandi
- nemos

modules to load probably:
- cuda

Probably default to installation on the cluster and accessing notebook through JupyterHub (https://wiki.flatironinstitute.org/SCC/JupyterHub). Outline (to be made more specific)
- recommendation to create virtual environment using system packages to avoid installing things that already exist

```
module load python 
VENVDIR=/path/to/wherever/you/want/to/store/your/venvs
python -m venv --system-site-packages $VENVDIR/name-of-your-venv
source $VENVDIR/name-of-your-venv/bin/activate
```
- Install above additional packages via pip
- Create jupyter kernel
```
module load jupyter-kernels
python -m make-custom-kernel mykernel
```

+++

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

+++

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
import os
import pynwb as nwb
import pynapple as nap
import pandas as pd

fpath = "/mnt/home/neurostatslab/ceph/000363/sub-441666"
fname = "sub-441666_ses-20190513T144253_behavior+ecephys+ogen.nwb"

# # os.path.join(fpath,fname)
# io = nwb.NWBHDF5IO(os.path.join(fpath,fname))
# nwbfile = io.read()
```

```{code-cell} ipython3
nwbfile
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
