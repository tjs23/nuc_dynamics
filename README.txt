NucDynamics genome/chromosome structure calculation scripts
-----------------------------------------------------------

This repository contains an ipython notebook script which demonstrates the
calculation of genome structures from single-cell Hi-C chromosome contact data
useing a simulated annealing partile dynamics protocol.

The basic operation of the particle dynamics is demonstrated by the
"anneal_structure" function. However, the full genome structure calculation,
including the derivation of distance restraints from contact data is provided by
the "anneal_genome" function.

The separate funtions and their test code operate within the larger script which
also contains all of the necessary utility functions (often written in Cython
for speed).


Installation
------------

After following the installation instructions, the notebook can be run with the
command jupyter-notebook in the repository folder.


Python Packages
---------------

The scripts depend on the NumPy and Cython packages. These are all available
through pip, and in most distributions' package managers. To view the notebook,
a web browser is required.


Jupyter
-------

Jupyter can be installed using pip install jupyter, for either Python3 and
Python2. The notebook is written in Python2, so the Python2 Jupyter kernel must
be installed:

  pip2 install ipykernel followed by python2 -m ipykernel install.

Note: it might be necessary to explicitly select the Python2 kernel from the
Kernel menu when running the notebook.

All scripts will be migrated to Python 3 in due course.


Data Files
----------

The script requires the demonstration data (available as a .tar.gz archive) to
be extracted into the same directory as the notebook script, but otherwise the
material is self-contained. 


Output
------

The result of the structure calculation is output in PDB format which may be
viewed in molecular graphics software such as PyMol.
