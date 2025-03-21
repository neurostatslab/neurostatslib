from functools import partial
from .loaders import dandi_downloader, load_nwb, load_mat, nap_load


DATA_REGISTRY = {
    "mesoscale_activity": {
        "sub-480928_ses-20210129T132207_behavior+ecephys+ogen.nwb": "3526a812f126fe09205c4ef5592974cce78bab5625f3eacf680165e35d56b443",
    },
    "perceptual_straightening": {
        "ay5_u002_image_sequences.mat": "c1b8a03e624a1e79b6c8c77fb3f9d83cd6fc9ee364f5ed334883bbc81c38ca0f",
        "stim_info.mat": "a7880cd0a0321d72c82f0639078aa017b9249a2bd90320c19182cd0ee34de890",
        "stim_matrix.mat": "910f4ac5a5a8b2ffd6ed165a9cd50260663500cd17ed69a547bca1f1ae3290fb",
    },
    "place_cells": {
        "Achilles_10252013_EEG.nwb": "a97a69d231e7e91c07e24890225f8fe4636bac054de50345551f32fc46b9efdd",
    },
}

DATA_URLS = {
    "mesoscale_activity": {
        "sub-480928_ses-20210129T132207_behavior+ecephys+ogen.nwb": "https://api.dandiarchive.org/api/assets/3d142f75-f3c0-4106-9533-710d26f12b02/download/",
    },
    "perceptual_straightening": {
        "ay5_u002_image_sequences.mat": "https://osf.io/9kbnw/download",
        "stim_info.mat": "https://osf.io/gwtcs/download",
        "stim_matrix.mat": "https://osf.io/bh6mu/download",
    },
    "place_cells": {
        "Achilles_10252013_EEG.nwb": "https://osf.io/2dfvp/download",
    },
}


DATA_DOWNLOADER = {
    "mesoscale_activity": dandi_downloader,
    # "ibl_data": dandi_downloader,
    "perceptual_straightening": None,
    "place_cells": None,
}

DATA_LOADER = {
    "mesoscale_activity": load_nwb,
    # "ibl_data": load_nwb,
    "perceptual_straightening": partial(load_mat, file_name="perceptual_straightening"),
    "place_cells": nap_load,
}


NOTEBOOK_REGISTRY = {
    "place_cells.md": "69fcaf2fe0faf3b9ee46209b89261713191e782c2784bc2960cf98ee21e7f34d",
}
