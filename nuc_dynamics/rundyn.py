import numpy as np

from . import dyn_util
from .util.coords import get_random_coords, pack_chromo_coords, unpack_chromo_coords
from .util.parallel import parallel_split_job
from .util.const import MAX_CORES


def anneal_model(model_data, anneal_schedule, masses, radii, restraint_indices, restraint_dists,
                 ambiguity, temp, time_step, dyn_steps, repulse, n_rep_max):

    import gc
    
    m, model_coords = model_data

    # Anneal one model in parallel
    time_taken = 0.0

    if m == 0:
        printInterval = max(1, dyn_steps/2)
    else:
        printInterval = 0

    print('    starting model %d' % m)
    
    for temp, repulse in anneal_schedule:
        gc.collect() # Try to free some memory
        
        # Update coordinates for this temp
        
        try:    
            dt, n_rep_max = dyn_util.runDynamics(model_coords, masses, radii, restraint_indices, restraint_dists,
                                                 ambiguity, temp, time_step, dyn_steps, repulse, nRepMax=n_rep_max,
                                                 printInterval=printInterval)
        
        except Exception as err:
            return err
 
        n_rep_max = np.int32(1.05 * n_rep_max) # Base on num in prev cycle, plus a small overhead
        time_taken += dt

    # Center
    model_coords -= model_coords.mean(axis=0)

    print('    done model %d' % m)
    
    return model_coords
 

def anneal_genome(chromosomes, contact_dict, num_models, particle_size,
                  general_calc_params, anneal_params,
                  prev_seq_pos_dict=None, start_coords=None, num_cpu=MAX_CORES):
    """
    Use chromosome contact data to generate distance restraints and then
    apply a simulated annealing protocul to generate/refine coordinates.
    Starting coordinates may be random of from a previous (e.g. lower
    resolution) stage.
    """
    
    from numpy import random
    from math import log, exp, atan, pi
    
    random.seed(general_calc_params['random_seed'])
    particle_size = np.int32(particle_size)

    # Calculate distance restrains from contact data     
    restraint_dict, seq_pos_dict = dyn_util.calc_restraints(chromosomes, contact_dict, particle_size,
        scale=1.0, exponent=general_calc_params['dist_power_law'],
        lower=general_calc_params['contact_dist_lower'],
        upper=general_calc_params['contact_dist_upper'])

    # Concatenate chromosomal data into a single array of particle restraints
    # for structure calculation. Add backbone restraints between seq. adjasent particles.
    restraint_indices, restraint_dists = dyn_util.concatenate_restraints(
        restraint_dict, seq_pos_dict, particle_size,
        general_calc_params['backbone_dist_lower'],
        general_calc_params['backbone_dist_upper'])

    # Setup starting structure
    if (start_coords is None) or (prev_seq_pos_dict is None):
        coords = get_random_coords(seq_pos_dict, chromosomes, num_models,
                                   general_calc_params['random_radius'])
        
        num_coords = coords.shape[1]
                    
    else:
        # Convert starting coord dict into single array
        coords = pack_chromo_coords(start_coords, chromosomes)
        num_coords = sum([len(seq_pos_dict[c]) for c in chromosomes])
            
        if coords.shape[1] != num_coords: # Change of particle_size
            interp_coords = np.empty((num_models, num_coords, 3))
            
            for m in range(num_models): # Starting coords interpolated from previous particle positions
                interp_coords[m] = dyn_util.getInterpolatedCoords(chromosomes, seq_pos_dict, prev_seq_pos_dict, coords[m])
            
            coords = interp_coords
            
    # Equal unit masses and radii for all particles
    masses = np.ones(num_coords, float)
    radii = np.ones(num_coords, float)
    
    # Ambiguiity strides not used here, so set to 1
    num_restraints = len(restraint_indices)
    ambiguity = np.ones(num_restraints, np.int32)
            
    # Below will be set to restrict memory allocation in the repusion list
    # (otherwise all vs all can be huge)
    n_rep_max = np.int32(0)
    
    # Annealing parameters
    temp_start = anneal_params['temp_start']
    temp_end = anneal_params['temp_end']
    temp_steps = anneal_params['temp_steps']
    
    # Setup annealig schedule: setup temps and repulsive terms
    adj = 1.0 / atan(10.0)
    decay = log(temp_start/temp_end)        
    anneal_schedule = []
    
    for step in range(temp_steps):
        frac = step/float(temp_steps)
    
        # exponential temp decay
        temp = temp_start * exp(-decay*frac)
    
        # sigmoidal repusion scheme
        repulse = 0.5 + adj * atan(frac*20.0-10) / pi 
        
        anneal_schedule.append((temp, repulse))    
            
    # Paricle dynamics parameters
    # (these need not be fixed for all stages, but are for simplicity)        
    dyn_steps = anneal_params['dynamics_steps']
    time_step = anneal_params['time_step']
         
    # Update coordinates in the annealing schedule which is applied to each model in parallel
    common_args = [anneal_schedule, masses, radii, restraint_indices, restraint_dists,
                   ambiguity, temp, time_step, dyn_steps, repulse, n_rep_max]
    
    task_data = [(m, coords[m]) for m in range(len(coords))]
    
    coords = parallel_split_job(anneal_model, task_data, common_args, num_cpu, collect_output=True)
    coords = np.array(coords)
    
    # Convert from single coord array to dict keyed by chromosome
    coords_dict = unpack_chromo_coords(coords, chromosomes, seq_pos_dict)
    
    return coords_dict, seq_pos_dict
