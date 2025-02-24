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

# Perceptual straightening

```{code-cell} ipython3
import pynacollada as nac
# load pynacollada data set
data = nac.load_data("perceptual_straightening")
data
```

```{code-cell} ipython3
data.ay5_u002_image_sequences.naturalStruct.sortedResponses
```

```{code-cell} ipython3
data.ay5_u002_image_sequences.naturalStruct.sortedResponses[:]
```