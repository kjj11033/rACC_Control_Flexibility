# Adaptive Drift-Diffusion Model (Adaptive DDM)

This repository contains the custom code and implementation of the Adaptive Drift-Diffusion Model (Adaptive DDM).

The model extends the standard Drift-Diffusion Model (DDM) implemented in the HDDM toolbox by incorporating a reinforcement-learning-based estimate of conflict probability (CP) to dynamically modulate the drift rate.

## System Requirements

* Operating System: Linux, macOS, or Windows.
* Hardware: Standard desktop computer.
* Software Dependencies:
    * Python 3.7+
    * HDDM (v0.9.8)
    * NumPy, SciPy, Pandas, Tqdm, Seaborn, Matplotlib
    * Cython (for compiling custom model extensions)
    * A C/C++ compiler (e.g., GCC on Linux, Clang on macOS, Visual Studio Build Tools on Windows)

## Installation Guide

The Adaptive DDM requires modifying the core HDDM library to register the custom likelihood functions (wfpt_addm). Please follow these steps carefully.

Estimated Install Time: ~20 minutes.

### 1. Install Base Dependencies
Please refer to the official HDDM documentation to install the base HDDM package:
https://hddm.readthedocs.io/en/latest/

We used HDDM version 0.9.8. In addition to HDDM, ensure you have the following packages installed in your environment: cython, pandas, matplotlib, seaborn, and tqdm.

### 2. Locate Your HDDM Installation
Find where HDDM is installed in your environment by running this command in Python:

python -c "import hddm; print(hddm.__path__)"
# Example output: /your/path/to/site-packages/hddm

Note: We will refer to the parent folder of hddm (e.g., /your/path/to/site-packages/) as SITE_PACKAGES.

### 3. Copy Custom Model Files
Move the provided custom model files from this repository into your HDDM library structure:

1. Copy integrate.pxi to SITE_PACKAGES/
2. Copy wfpt_addm.pyx and wfpt_addm_ics.pyx to SITE_PACKAGES/
3. Copy hddm_addm.py and hddm_addm_ics.py to SITE_PACKAGES/hddm/models/

### 4. Compile the Custom Extensions
You must Cythonize the .pyx files to make them executable. Create a script named setup_custom_models.py in the SITE_PACKAGES directory with the following content:

# setup_custom_models.py
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

# Ensure we are pointing to the correct include directories
setup(
    include_dirs = [np.get_include()], 
    ext_modules = cythonize(['wfpt_addm.pyx', 'wfpt_addm_ics.pyx'])
)

Run the compilation from your terminal inside the SITE_PACKAGES folder:

python setup_custom_models.py build_ext --inplace

### 5. Register Models in __init__.py
Open SITE_PACKAGES/hddm/models/__init__.py in a text editor.

1. Add these imports:
from .hddm_addm import HDDM_addm
from .hddm_addm_ics import HDDM_addm_ics

2. Add the model names to the __all__ list:
__all__ = [..., "HDDM_addm", "HDDM_addm_ics"]

---

## Demo & Usage

### Data Format
Your input CSV must contain the following columns:
* subj_idx: Subject identifier (integer)
* rt: Reaction time in seconds
* response: 1 (correct) or 0 (incorrect)
* conflict: 1 (Conflict trial) or 0 (Non-Conflict trial)
* starting_trial: 1 for the first trial of a block, 0 otherwise
* ics: (Optional) 1 for Stim-ON, 0 for Stim-OFF

### Running the Model
See Run_adaptive_DDM_example.ipynb for a step-by-step demonstration.

* Full Analysis Runtime: ~4-6 hours for 30,000 samples on the full dataset (N=21 subjects).
* Quick Demo: To verify installation quickly, reduce nsample to 1000 in the notebook. This will run in <10 minutes.

---

## License & Citation

### Code License
This custom code is released under the MIT License.

### Acknowledgments
This project utilizes the HDDM toolbox (Wiecki et al., 2013). We gratefully acknowledge the developers of HDDM for their foundational work.

### Citation
If you use this code in your research, please cite:
Kim, J., & Widge, A. S. (2025). Cingulate-centered flexible control: physiologic correlates and enhancement by internal capsule stimulation. bioRxiv, 2025.10.15.682151. https://doi.org/10.1101/2025.10.15.682151
Wiecki TV, Sofer I, Frank MJ (2013). HDDM: Hierarchical Bayesian estimation of the Drift-Diffusion Model in Python. Frontiers in Neuroinformatics 7: 14.


