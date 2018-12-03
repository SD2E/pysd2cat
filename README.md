Introduction
==========

The `pysd2cat` circuit analysis tool analyzes flow cytometry data to perform the following tasks:

* Predict Live and Dead cells
* Compute the accuracy of a circuit wrt. a GFP threshold


Quick Start:
=========
Currently, `pysd2cat` runs on the TACC infrastructure so that it may have fast access to data. It assumes a root directory (containing data files) exists on the host at: `/work/projects/SD2E-Community/prod/data/uploads/`.

* Clone the repo: `https://gitlab.sd2e.org/dbryce/pysd2cat.git`
* `cd pysd2cat`
* `python setup.py install`

To run an example analysis script, run:
* `python src/pysd2cat/analysis/live_dead_analysis.py`

Code Layout:
===========

The source code is divided into subdirectories, as follows:

* `src/data`: routines to acquire data and metadata
* `src/analysis`: routines to analyze data
* `src/plot`: routines to plot data