#!/usr/bin/python
"""
.. module:: MontePython
   :synopsis: Main module
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>

Monte Python, a Monte Carlo Markov Chain code (with Class!)
"""
import sys
import warnings

import io_mp       # all the input/output mechanisms
from run import run


# -----------------MAIN-CALL---------------------------------------------
if __name__ == '__main__':
    # Default action when facing a warning is being remapped to a custom one
    warnings.showwarning = io_mp.warning_message

    # MPI is tested for, and if a different than one number of cores is found,
    # it runs mpi_run instead of a simple run.
    mpi_asked = False
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        nprocs = comm.Get_size()
        if nprocs > 1:
            mpi_asked = True
    # If the ImportError is raised, it only means that the Python wrapper for
    # MPI is not installed - the code should simply proceed with a non-parallel
    # execution.
    except ImportError:
        pass

    if mpi_asked:
        # This import has to be there in case MPI is not installed
        from run import mpi_run
        sys.exit(mpi_run())
    else:
        sys.exit(run())