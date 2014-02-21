"""
.. module:: nested_sampling
    :synopsis: Interface the MultiNest program with Monte Python

This implementation relies heavily on the existing Python wrapper for
MultiNest, called PyMultinest, written by Johannes Buchner, and available `at
this address <https://github.com/JohannesBuchner/PyMultiNest>`_ .

The main routine, :func:`run`, truly interfaces the two codes. It takes for
input the cosmological module, data and command line. It then defines
internally two functions, :func:`prior() <nested_sampling.prior>` and
:func:`loglike` that will serve as input for the run function of PyMultiNest.

.. moduleauthor:: Jesus Torrado <torradocacho@lorentz.leidenuniv.nl>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
"""
from pymultinest import run as nested_run
import numpy as np
import os
import io_mp
import sampler
import warnings

# Data on file names and MultiNest options, that may be called by other modules

# MultiNest subfolder and name separator
NS_subfolder    = 'NS'
NS_separator    = '-'
# MultiNest file names ending, i.e. after the defined 'base_name'
name_rejected   = NS_separator + 'ev.dat'                 # rejected points
name_post       = NS_separator + '.txt'                   # accepted points
name_post_sep   = NS_separator + 'post_separate.dat'      # accepted points separated by '\n\n'
name_post_equal = NS_separator + 'post_equal_weights.dat' # some acc. points, same sample prob.
name_stats      = NS_separator + 'stats.dat'              # summarized information, explained
name_summary    = NS_separator + 'summary.txt'            # summarized information
# New files
name_paramnames = '.paramnames'            # in the NS/ subfolder
name_arguments  = '.arguments'             # in the NS/ subfolder
name_chain_acc  = 'chain_NS__accepted.txt' # in the chain root folder
name_chain_rej  = 'chain_NS__rejected.txt' # in the chain root folder
# Log.param name (ideally, we should import this one from somewhere else)
name_logparam = 'log.param'

# Multinest option prefix
NS_prefix       = 'NS_'
# User-defined arguments of PyMultiNest, and 'argparse' keywords
# First: basic string -> bool type conversion:
str2bool = lambda s: True if s.lower() == 'true' else False
NS_user_arguments = {
    # General sampling options
    'n_live_points':
        {'metavar': 'Number of live samples',
         'type': int},
    'importance_nested_sampling':
        {'metavar': 'True or False',
         'type': str2bool},
    'sampling_efficiency':
        {'metavar': 'Sampling efficiency',
         'type': float},
    'const_efficiency_mode':
        {'metavar': 'True or False',
         'type': str2bool},
    'seed':
        {'metavar': 'Random seed',
         'type': int},
    'log_zero':
        {'metavar': 'Min. log-evidence to consider',
         'type': float},
    'n_iter_before_update':
        {'metavar': 'Number of iterations between updates',
         'type': int},
    # Ending conditions
    'evidence_tolerance':
        {'metavar': 'Evidence tolerance',
         'type': float},
    'max_iter':
        {'metavar': 'Max. number of iterations',
         'type': int},
    # Multimodal sampling
    'multimodal':
        {'metavar': 'True or False',
         'type': str2bool},
    'max_modes':
        {'metavar': 'Max. number of modes to consider',
         'type': int},
    'mode_tolerance':
        {'metavar': 'Min. value of the log-evidence for a mode to be considered',
         'type': float},
    'clustering_params':
        {'metavar': 'Parameters to be used for mode separation',
         'type': str,
         'nargs': '+'}
    }
# Automatically-defined arguments of PyMultiNest, type specified
NS_auto_arguments = {
    'n_dims':   {'type': int},
    'n_params': {'type': int},
    'verbose':  {'type': str2bool},
    'outputfiles_basename': {'type': str}
    }


def run(cosmo, data, command_line):
    """
    Main call to prepare the information for the MultiNest run, and to actually
    run the MultiNest sampler.

    Note the unusual set-up here, with the two following functions, `prior` and
    `loglike` having their docstrings written in the encompassing function.
    This trick was necessary as MultiNest required these two functions to be
    defined with a given number of parameters, so we could not add `data`. By
    defining them inside the run function, this problem was by-passed.

    .. function:: prior

        Generate the prior function for MultiNest

        It should transform the input unit cube into the parameter cube. This
        function actually wraps the method :func:`map_from_unit_interval()
        <prior.Prior.map_from_unit_interval>` of the class :class:`Prior
        <prior.Prior>`.

        :Parameters:
            **cube** (`array`) - Contains the current point in unit parameter
                space that has been selected within the MultiNest part.
            **ndim** (`int`) - Number of varying parameters
            **nparams** (`int`) - Total number of parameters, including the
                derived ones (not used, so hidden in `*args`)


    .. function:: loglike

        Generate the Likelihood function for MultiNest

        :Parameters:
            **cube** (`array`) - Contains the current point in the correct
                parameter space after transformation from :func:`prior`.
            **ndim** (`int`) - Number of varying parameters
            **nparams** (`int`) - Total number of parameters, including the
                derived ones (not used, so hidden in `*args`)

    """
    # Convenience variables
    varying_param_names = data.get_mcmc_parameters(['varying'])
    derived_param_names = data.get_mcmc_parameters(['derived'])

    # Check that all the priors are flat and that all the parameters are bound
    if not(all(data.mcmc_parameters[name]['prior'].prior_type == 'flat'
               for name in varying_param_names)):
        raise io_mp.ConfigurationError(
            'Nested Sampling with MultiNest is only possible with flat ' +
            'priors. Sorry!')
    if not(all(data.mcmc_parameters[name]['prior'].is_bound()
               for name in varying_param_names)):
        raise io_mp.ConfigurationError(
            'Nested Sampling with MultiNest is only possible for bound ' +
            'parameters. Set reasonable bounds for them in the ".param"' +
            'file.')

    # If absent, create the sub-folder NS
    NS_folder = os.path.join(command_line.folder, NS_subfolder)
    if not os.path.exists(NS_folder):
        os.makedirs(NS_folder)

    # Use chain name as a base name for MultiNest files
    chain_name = [a for a in command_line.folder.split(os.path.sep) if a][-1]
    base_name = os.path.join(NS_folder, chain_name)

    # Prepare arguments for PyMultiNest
    # -- Automatic arguments
    data.NS_arguments['n_dims']   =  len(varying_param_names)
    data.NS_arguments['n_params'] = (len(varying_param_names) +
                                      len(derived_param_names))
    data.NS_arguments['verbose']  = True
    data.NS_arguments['outputfiles_basename'] = base_name + NS_separator
    # -- User-defined arguments
    for arg in NS_user_arguments:
        value = getattr(command_line, NS_prefix+arg)
        # Special case: clustering parameters
        if arg == 'clustering_params':
            clustering_param_names = value if value != -1 else []
            continue
        # Rest of the cases
        if value != -1:
            data.NS_arguments[arg] = value
        # else: don't define them -> use PyMultiNest default value

    # Clustering parameters -- reordering to put them first
    NS_param_names = []
    if clustering_param_names:
        data.NS_arguments['n_clustering_params'] = len(clustering_param_names)
        for param in clustering_param_names:
            if not param in varying_param_names:
                raise io_mp.ConfigurationError(
                'The requested clustering parameter "%s"'%param+
                ' was not found in your ".param" file. Pick a valid one.')
            NS_param_names.append(param)
    for param in varying_param_names:
        if not param in NS_param_names:
            NS_param_names.append(param)       
        
    # Caveat: multi-modal sampling OFF by default; if requested, INS disabled
    try:
        if data.NS_arguments['multimodal']:
            data.NS_arguments['importance_nested_sampling'] = False
            warnings.warn('Multi-modal sampling has been requested, '+
                          'so Importance Nested Sampling has been disabled')
    except KeyError:
        data.NS_arguments['multimodal'] = False

    # Write the MultiNest arguments and parameter ordering
    with open(base_name+name_arguments, 'w') as afile:
        for arg in data.NS_arguments:
            if arg != 'n_clustering_params':
                afile.write(' = '.join([str(arg), str(data.NS_arguments[arg])]))
            else:
                afile.write('clustering_params = '+
                            ' '.join(clustering_param_names))
            afile.write('\n')
    with open(base_name+name_paramnames, 'w') as pfile:
        pfile.write('\n'.join(NS_param_names+derived_param_names))

    # Function giving the prior probability
    def prior(cube, ndim, *args):
        """
        Please see the encompassing function docstring

        """
        for i, name in zip(range(ndim), NS_param_names):
            cube[i] = data.mcmc_parameters[name]['prior']\
                .map_from_unit_interval(cube[i])

    # Function giving the likelihood probability            
    def loglike(cube, ndim, *args):
        """
        Please see the encompassing function docstring

        """
        # Updates values: cube --> data
        for i, name in zip(range(ndim), NS_param_names):
            data.mcmc_parameters[name]['current'] = cube[i]
        # Propagate the information towards the cosmo arguments
        data.update_cosmo_arguments()
        lkl = sampler.compute_lkl(cosmo, data)
        for i, name in enumerate(derived_param_names):
            cube[ndim+i] = data.mcmc_parameters[name]['current']
        return lkl

    # Launch MultiNest, and recover the output code
    output = nested_run(loglike, prior, **data.NS_arguments)
    
    # Assuming this worked, i.e. if output is `None`,
    # state it and suggest the user to analyse the output.
    if output is None:
        warnings.warn('The sampling with MultiNest is done.\n' +
                      'You can now analyse the output calling MontePython ' +
                      ' with the -info flag in the chain_name/NS subfolder.')

        
def from_NS_output_to_chains(folder):
    """
    Translate the output of MultiNest into readable output for Monte Python

    This routine will be called by the analyze modules.

    If mode separation has been performed (i.e., multimodal=True), it creates
    'mode_#' subfolders containing a chain file with the corresponding samples
    and a 'log.param' file in which the starting point is the best fit of the
    nested sampling, and the same for the sigma. The minimum and maximum value
    are cropped to the extent of the modes in the case of the parameters used
    for the mode separation, and preserved in the rest.

    The mono-modal case is treated as a special case of the multi-modal one.

    """
    chain_name = [a for a in folder.split(os.path.sep) if a][-2]
    base_name = os.path.join(folder, chain_name)

    # Read the arguments of the NS run
    # This file is intended to be machine generated: no "#" ignored or tests done
    NS_arguments = {}
    with open(base_name+name_arguments, 'r') as afile:
        for line in afile:
            arg   = line.split('=')[0].strip()
            value = line.split('=')[1].strip()
            arg_type = (NS_user_arguments[arg]['type']
                        if arg in NS_user_arguments else
                        NS_auto_arguments[arg]['type'])
            value = arg_type(value)
            if arg == 'clustering_params':
                value = [a.strip() for a in value.split()]
            NS_arguments[arg] = value
    # Read parameters order
    NS_param_names = np.loadtxt(base_name+name_paramnames, dtype='str').tolist()

    # Open the 'stats.dat' file to see what happened and retrieve some info
    stats_file = open(base_name+name_stats, 'r')
    lines = stats_file.readlines()
    stats_file.close()
    # Mode-separated info
    multimodal = NS_arguments.get('multimodal')
    i = 0
    n_modes = 0
    stats_mode_lines = {0:[]}
    for line in lines:
        if 'Nested Sampling Global Log-Evidence' in line:
            global_logZ, global_logZ_err = [float(a.strip()) for a in
                                            line.split(':')[1].split('+/-')]
        if 'Total Modes Found' in line:
            n_modes = int(line.split(':')[1].strip())
        if line[:4] == 'Mode':
            i += 1
            stats_mode_lines[i] = []
        # This stores the info of each mode i>1 in stats_mode_lines[i]
        #    and in i=0 the lines previous to the modes, in the multi-modal case
        #    or the info of the only mode, in the mono-modal case
        stats_mode_lines[i].append(line)
    assert n_modes == max(stats_mode_lines.keys()), (
        'Something is wrong... (strange error n.1)')

    # Prepare the accepted-points file -- modes are separated by 2 line breaks
    accepted_name = base_name + (name_post_sep if multimodal else name_post)
    with open(accepted_name, 'r') as accepted_file:
        mode_lines = [a for a in ''.join(accepted_file.readlines()).split('\n\n')
                      if a != '']

    if multimodal:
        assert len(mode_lines) == n_modes, 'Something is wrong... (strange error n.2a)'
    else:
        assert len(mode_lines) == 1, 'Something is wrong... (strange error n.2b)'

# TODO: prepare total and rejected chain
  
    # Preparing log.param files of modes
    with open(os.path.join(chain_name, name_logparam), 'r') as log_file:
        log_lines = log_file.readlines()
    # Number of the lines to be changed
    varying_param_names = data.get_mcmc_parameters(['varying'])
    param_lines = {}
    pre, pos = 'data.parameters[', ']'
    for i, line in enumerate(log_lines):
        if pre in line:
            if line.strip()[0] == '#':
                continue
            param_name = line.split('=')[0][line.find(pre)+len(pre):
                                            line.find(pos)]
            param_name = param_name.replace('"','').replace("'",'').strip()
            if param_name in varying_param_names:
                param_lines[param_name] = i

                
    # Parameters to cut: clustering_params, if exists, otherwise varying_params
    cut_params = data.NS_parameters.get('n_clustering_params')
    if cut_params and multimodal:
        cut_param_names = varying_param_names[:cut_params]
    elif multimodal:
        cut_param_names = varying_param_names
    # mono-modal case
    else:
        cut_param_names = []




## MultiNest file names ending, i.e. after the defined 'base_name'
#name_rejected   = '-ev.dat'                 # rejected points
#name_post       = '.txt'                   # accepted points
#name_post_sep   = '-post_separate.dat'      # accepted points, separated by '\n\n'
#name_post_equal = '-post_equal_weights.dat' # some acc. points, same sample prob.
#name_stats      = '-stats.dat'              # summarized information, explained
#name_summary    = '-summary.txt'            # summarized information
# New files
#name_paramnames = '.paramnames'            # in the NS/ subfolder
#name_chain_acc  = 'chain_NS__accepted.txt'
#name_chain_rej  = 'chain_NS__rejected.txt'

#accepted_chain_name = 'chain_NS__accepted.txt'

# SEGUIR DESDE AQUI



        
    # Process each mode:
    ini = 1 if multimodal else 0
    for i in range(ini, 1+n_modes):
        # Create subfolder
        if multimodal:
            mode_subfolder = 'mode_'+str(i).zfill(len(str(n_modes)))
        else:
            mode_subfolder = ''
        mode_subfolder = os.path.join(command_line.folder, mode_subfolder)
        if not os.path.exists(mode_subfolder):
            os.makedirs(mode_subfolder)

        # Add ACCEPTED points
        mode_data = np.array(mode_lines[i].split(), dtype='float64')
        columns = 2+data.NS_parameters['n_params']
        mode_data = mode_data.reshape([mode_data.shape[0]/columns, columns])
        # Rearrange: sample-prob | -2*loglik | params
        #       ---> sample-prob |   -loglik | params
        mode_data[:, 1] = mode_data[: ,1] / 2.
        np.savetxt(os.path.join(mode_subfolder, accepted_chain_name),
                   mode_data, fmt='%.6e')

        # Get the necessary info of the parameters:
        #  -- max_posterior (MAP), sigma  <---  stats.dat file
        for j, line in enumerate(stats_mode_lines[i]):
            if 'Sigma' in line:
                line_sigma = j+1
            if 'MAP' in line:
                line_MAP = j+2
        MAPs   = {}
        sigmas = {}
        for j, param in enumerate(varying_param_names):
            n, MAP = stats_mode_lines[i][line_MAP+j].split()
            assert int(n) == j+1,  'Something is wrong... (strange error n.3)'
            MAPs[param] = MAP
            n, mean, sigma = stats_mode_lines[i][line_sigma+j].split()
            assert int(n) == j+1,  'Something is wrong... (strange error n.4)'
            sigmas[param] = sigma
        #  -- minimum rectangle containing the mode (only clustering params)
        mins = {}
        maxs = {}
        for j, param in enumerate(varying_param_names):
            if param in cut_param_names:
                mins[param] = min(mode_data[:, 2+j])
                maxs[param] = max(mode_data[:, 2+j])
            else:
                mins[param] = data.mcmc_parameters[param]['initial'][1]
                maxs[param] = data.mcmc_parameters[param]['initial'][2]
        # Create the log.param file
        for param in varying_param_names:
            line = pre+"'"+param+"'] = ["
            values = [MAPs[param], '%.6e'%mins[param], '%.6e'%maxs[param],
                      sigmas[param], '%e'%data.mcmc_parameters[param]['scale'],
                      "'"+data.mcmc_parameters[param]['role']+"'"]
            line += ', '.join(values) + ']\n'
            log_lines[param_lines[param]] = line

        # TODO: HANDLE SCALING!!!!

        with open(os.path.join(mode_subfolder, 'log.param'), 'w') as log_file:
            log_file.writelines(log_lines)

        # TODO: USE POINTS FROM TOTAL AND REJECTED SAMPLE???




### THE NEXT FUNCTION IS CURRENTLY NOT USED:
def from_NS_output_to_chains_OLD(folder, base_name):
    """
    Translate the output of MultiNest into readable output for Monte Python

    This routine will be called after the MultiNest run has been successfully
    completed.

    """
    # First, take care of post_equal_weights (accepted points)
    accepted_chain = os.path.join(folder,
                                  'chain_NS__accepted.txt')
    rejected_chain = os.path.join(folder,
                                  'chain_NS__rejected.txt')

    # creating chain of accepted points (straightforward reshuffling of
    # columns)
    with open(base_name+'post_equal_weights.dat', 'r') as input_file:
        output_file = open(accepted_chain, 'w')
        array = np.loadtxt(input_file)
        output_array = np.ones((np.shape(array)[0], np.shape(array)[1]+1))
        output_array[:, 1] = -array[:, -1]
        output_array[:, 2:] = array[:, :-1]
        np.savetxt(
            output_file, output_array,
            fmt='%i '+' '.join(['%.6e' for _ in
                               range(np.shape(array)[1])]))
        output_file.close()

    # Extracting log evidence
    with open(base_name+'stats.dat') as input_file:
        lines = [line for line in input_file if 'Global Log-Evidence' in line]
        if len(lines) > 1:
            lines = [line for line in lines if 'Importance' in line]
        log_evidence = float(lines[0].split(':')[1].split('+/-')[0])

    # Creating chain from rejected points, with some interpretation of the
    # weight associated to each point arXiv:0809.3437 sec 3
    with open(base_name+'ev.dat', 'r') as input_file:
        output = open(rejected_chain, 'w')
        array = np.loadtxt(input_file)
        output_array = np.zeros((np.shape(array)[0], np.shape(array)[1]-1))
        output_array[:, 0] = np.exp(array[:, -3]+array[:, -2]-log_evidence)
        output_array[:, 0] *= np.sum(output_array[:, 0])*np.shape(array)[0]
        output_array[:, 1] = -array[:, -3]
        output_array[:, 2:] = array[:, :-3]
        np.savetxt(
            output, output_array,
            fmt=' '.join(['%.6e' for _ in
                         range(np.shape(output_array)[1])]))
        output.close()

