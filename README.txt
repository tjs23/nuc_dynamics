NucDynamics
-----------

NucDynamics is a Python/Cython program for the calculation of genome structures
from single-cell Hi-C chromosome contact data using a simulated annealing
particle dynamics protocol. This software takes NCC format contact data (as
output by NucProcess) and creates PDB (ProteinDataBank) format files. The result
of the structure calculation is output in PDB format so that it may be viewed in
molecular graphics software such as PyMol. Further output formats will be
supported in the near future.

To run NucDynamics issue the 'nuc_dynamics' command line followed by the
name/location of an input NCC format contact file. Various options may be
specfied on the command line using flags that are prefixed with '-'. A full
listing of these is given below, but commonly -o (the output coordinate file),
-m (the number of conformational models to generate) and -s (the particle size
or sizes to use) will be specified.

Parameters relating to the restraint distances are not normally changed when
calculating interphase genome structures. However, the annealing stage
temperatures and the number of temperature and dynamics steps may be adjusted
according to the amount and/or quality of the single-cell Hi-C contacts. The
default parameters are generally suitable for good data sets with at least
40,000 contacts but better results may be obtained with longer and more gentle
annealing, i.e. more temperature steps and dynamics steps.

It should be noted that obtaining high-quality whole-genome structures (with a
tight conformational bundle) is dependant on both having a good number of
contacts and having a reasonable proportion of inter-chromosomal (trans)
contacts. As a rough guide, around 40,000 total contacts with 2000 trans
contacts is about the lower limit for calculating resonable structures
(intra-bundle RMSD < 2 radii) at a 100 kb particle size. Though, having a higher
proportion of trans contacts can make up for shortfalls in the total count.

The quality of the output structure bundles naturally depends on the final
sequence resolution that is chosen, as specified by the particle size (-s
option). Though the default paramaters calculate to a final size of 100 kb, for
more sparse datasets larger sizes often give good results. Using finer sizes can
provide more detail, but will increase the number of particles that are not
restrained by at least one Hi-C derived contact, and thus the precision (RMSD)
at the particle scale will decrease. Though, with >100,000 contacts particle
sizes of 25 kb are reasonable. However, finer resolutions will naturally
increase memory requirements and calculation time. 

Multiple particle sizes can be specified so that the genome structure
calculation uses a hierarchical protocol, calculating a low resolution structure
first and then basing the next, finer resolution stage on the output of the
previous stage. Hierarchical particle sizes need not be used, but they make the
calculation more robust. 


Python Module Requirements
---------------------------

This software uses Python version 2 or 3 and requires that the Numpy and Cython
packages are installed and available to the Python version that runs
NucDynamics.

These modules are available in bundled Python packages like Anaconda or Canopy,
in most Linux distributions' package managers or can be installed on most
UNIX-like systems using pip:

  pip install numpy
  pip install cython


Installation
------------

NucDynamics does not require installation as such and may be run directly from
its download location, though all the component files must reside in the same
directory.

When first run, NucDynamics will attempt to compile the modules written in
Cython. A re-compilation may be forced by deleting the .so and .c files that
result from the compilation. The Cython code may also be compiled indepenently
using the setup_cython.py script as follows:

  python setup_cython.py build_ext --inplace


Running NucDynamics
-------------------

Typical use, generating 10 conformational models:

  nuc_dynamics example_chromo_data/Cell_1_contacts.ncc -m 10


Specifying the particle sizes (8 Mb, 2 Mb, 1 Mb, 500 kb) and an output file
name:

  nuc_dynamics example_chromo_data/Cell_1_contacts.ncc -m 10 -o Cell_1.pdb -s 8 2 1 0.5


Example Data
------------

Example NCC format contact data to demonstrate NucDynamics is avaiable in the
example_chromo_data sub-directory, as a .tar.gz archive which must be extracted
before use. 


Command line options for nuc_dynamics
-------------------------------------

usage: nuc_dynamics [-h] [-o PDB_FILE] [-m NUM_MODELS]
                    [-s Mb_SIZE [Mb_SIZE ...]] [-iso Mb_SIZE] [-pow FLOAT]
                    [-lower DISTANCE] [-upper DISTANCE] [-bb_lower DISTANCE]
                    [-bb_upper DISTANCE] [-ran INT] [-rad DISTANCE]
                    [-hot TEMP_KELVIN] [-cold TEMP_KELVIN] [-temps NUM_STEPS]
                    [-dyns NUM_STEPS] [-time_step TIME_DELTA]
                    NCC_FILE

Single-cell Hi-C genome and chromosome structure calculation module for Nuc3D
and NucTools

positional arguments:
  NCC_FILE              Input NCC format file containing single-cell Hi-C
                        contact data, e.g. use the demo data at
                        example_chromo_data/Cell_1_contacts.ncc

optional arguments:
  -h, --help            show this help message and exit
  -o PDB_FILE           Optional name of PDB format output file for 3D
                        coordinates. If not set this will be auto-generated
                        from the input file name
  -m NUM_MODELS         Number of alternative conformations to generate from
                        repeat calculations with different random starting
                        coordinates: Default: 1
  -s Mb_SIZE [Mb_SIZE ...]
                        One or more sizes (Mb) for the hierarchical structure
                        calculation protocol (will be used in descending
                        order). Default: 8.0 4.0 2.0 0.4 0.2 0.1
  -iso Mb_SIZE          Contacts must be near another, within this (Mb)
                        separation threshold (at both ends) to be considered
                        supported: Default 2.0
  -pow FLOAT            Distance power law for combining multiple Hi-C
                        contacts between the same particles. Default: -0.33
  -lower DISTANCE       Lower limit for a contact-derived distance restraint,
                        as a fraction of the ideal distance. Default: 0.8
  -upper DISTANCE       Upper limit for a contact-derived distance restraint,
                        as a fraction of the ideal distance. Default: 1.2
  -bb_lower DISTANCE    Lower limit for sequential particle backbone
                        restraints, as a fraction of the ideal distance.
                        Default: 0.1
  -bb_upper DISTANCE    Upper limit for sequential particle backbone
                        restraints, as a fraction of the ideal distance.
                        Default: 1.1
  -ran INT              Seed for psuedo-random number generator
  -rad DISTANCE         Radius of sphere for random starting coordinates.
                        Default: 10.0
  -hot TEMP_KELVIN      Start annealing temperature in pseudo-Kelvin units.
                        Default: 5000
  -cold TEMP_KELVIN     End annealing temperature in pseudo-Kelvin units.
                        Default: 10
  -temps NUM_STEPS      Number of temperature steps in annealing protocol
                        between start and end temperature. Default: 500
  -dyns NUM_STEPS       Number of particle dynamics steps to apply at each
                        temperature in the annealing protocol. Default: 100
  -time_step TIME_DELTA
                        Simulation time step between re-calculation of
                        particle velocities. Default: 0.001

For further help on running this program please email tjs23@cam.ac.uk


Jupyter Script
--------------

In addition to the command-line tool a Jupyter notebook for Python 3 is provided
to illustrate how nuc_dynamics can be imported used within Python scripts.
Jupyter can be installed using:

  pip install jupyter

And from the directory containing nuc_dynamics the notebook can be started with:

  jupyter notebook

This notebook is has only been tested under Python version 3 and to run requires
the Cython code to be compiled (see Installation section above), to generate the
dyn_util.so file, and for all the modules to either be in the same directory or
on the PYTHONPATH.

 

