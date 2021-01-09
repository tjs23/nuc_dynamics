import numpy as np

from .util.const import N3D, PDB, FORMATS, MAX_CORES
from .util.contacts import remove_isolated_contacts, remove_violated_contacts
from .util import warn, critical, test_imports
from .io.loadtext import load_ncc_file, load_pairs_file, load_n3d_coords
from .io.export import export_coords
from .rundyn import anneal_genome


PROG_NAME = 'nuc_dynamics'
DESCRIPTION = 'Single-cell Hi-C genome and chromosome structure calculation module for Nuc3D and NucTools'


def calc_genome_structure(input_file_path, out_file_path, general_calc_params, anneal_params,
                          particle_sizes, num_models=5, isolation_threshold=2e6,
                          out_format=N3D, split_chromosome=False, num_cpu=MAX_CORES,
                          start_coords_path=None, save_intermediate=False):

    from time import time

    # Load single-cell Hi-C data from NCC contact file, as output from NucProcess
    if '.ncc' in os.path.split(input_file_path)[1]:
        chromosomes, chromo_limits, contact_dict = load_ncc_file(input_file_path)
    else:
        chromosomes, chromo_limits, contact_dict = load_pairs_file(input_file_path)

    # Only use contacts which are supported by others nearby in sequence, in the initial instance
    remove_isolated_contacts(contact_dict, threshold=isolation_threshold)

    # Initial coords will be random
    start_coords = None

    # Record particle positions from previous stages
    # so that coordinates can be interpolated to higher resolution
    prev_seq_pos = None

    if start_coords_path:
        prev_seq_pos, start_coords = load_n3d_coords(start_coords_path)
        if start_coords:
            chromo = next(iter(start_coords)) # picks out arbitrary chromosome
            num_models = len(start_coords[chromo])
        
    coords_dict = None
    particle_seq_pos = None
    particle_size = None

    for stage, particle_size in enumerate(particle_sizes):
 
            print("Running structure caculation stage %d (%d kb)" % (stage+1, (particle_size/1e3)))
 
            # Can remove large violations (noise contacts inconsistent with structure)
            # once we have a reasonable resolution structure
            
            if stage > 0:
                if particle_size < 0.5e6:
                        remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos,
                                                 particle_size, threshold=6.0)
                elif particle_size < 0.25e6:
                        remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos,
                                                 particle_size, threshold=5.0)
 
            coords_dict, particle_seq_pos = anneal_genome(chromosomes, contact_dict, num_models, particle_size,
                                                          general_calc_params, anneal_params,
                                                          prev_seq_pos, start_coords, num_cpu)
 
            if save_intermediate and stage < len(particle_sizes)-1:
                file_path = '%s_%d.%s' % (out_file_path[:-4], stage, out_file_path[-3:]) # DANGER: assumes that suffix is 3 chars
                export_coords(out_format, file_path, coords_dict, particle_seq_pos, particle_size)
                
            # Next stage based on previous stage's 3D coords
            # and their respective seq. positions
            start_coords = coords_dict
            prev_seq_pos = particle_seq_pos

    # Save final coords
    export_coords(out_format, out_file_path, coords_dict, particle_seq_pos, particle_size, split_chromosome)



def demo_calc_genome_structure():
    """
    Example of settings for a typical genome structure calculation from input single-cell
    Hi-C contacts stored in an NCC format file (as output from NucProcess)
    """
    
    ncc_file_path = 'example_chromo_data/Cell_1_contacts.ncc'
    save_path = 'example_chromo_data/Cell_1_structure.pdb'
    
    # Number of alternative conformations to generate from repeat calculations
    # with different random starting coordinates
    num_models = 2

    # Parameters to setup restraints and starting coords
    general_calc_params = {'dist_power_law':-0.33,
                           'contact_dist_lower':0.8, 'contact_dist_upper':1.2,
                           'backbone_dist_lower':0.1, 'backbone_dist_upper':1.1,
                           'random_seed':int(time()), 'random_radius':10.0}

    # Annealing & dyamics parameters: the same for all stages
    # (this is cautious, but not an absolute requirement)
    anneal_params = {'temp_start':5000.0, 'temp_end':10.0, 'temp_steps':500,
                     'dynamics_steps':100, 'time_step':0.001}

    # Hierarchical scale protocol
    particle_sizes = [8e6, 4e6, 2e6, 4e5, 2e5, 1e5]
    
    # Contacts must be clustered with another within this separation threshold
    # (at both ends) to be considered supported, i.e. not isolated
    isolation_threshold=2e6
    
    calc_genome_structure(ncc_file_path, save_path, general_calc_params, anneal_params,
                          particle_sizes, num_models, isolation_threshold, out_format='pdb')


test_imports()


if __name__ == '__main__':
    
    import os, sys
    from time import time
    from argparse import ArgumentParser
    
    epilog = 'For further help on running this program please email tjs23@cam.ac.uk'
    
    arg_parse = ArgumentParser(
        prog=PROG_NAME, description=DESCRIPTION, epilog=epilog, prefix_chars='-', add_help=True)

    arg_parse.add_argument(
        'input_file', nargs=1,
        metavar='INPUT_FILE',
        help='Input NCC/PAIRs format file containing single-cell Hi-C contact data, e.g. use the demo data at example_chromo_data/Cell_1_contacts.ncc')

    arg_parse.add_argument(
        '-o',
        metavar='OUT_FILE',
        help='Optional name of output file for 3D coordinates in N3D or PDB format (see -f option). If not set this will be auto-generated from the input file name')

    arg_parse.add_argument(
        '-save_intermediate',
        default=False,
        action='store_true',
        help='Write out intermediate coordinate files.')

    arg_parse.add_argument(
        '-start_coords_path', metavar='N3D_FILE',
        help='Initial 3D coordinates in N3D format. If set this will override -m flag.')

    arg_parse.add_argument(
        '-m', default=1, metavar='NUM_MODELS',
        type=int,
        help='Number of alternative conformations to generate from repeat calculations with different random starting coordinates: Default: 1')

    arg_parse.add_argument(
        '-f', metavar='OUT_FORMAT', default=N3D,
        help='File format for output 3D coordinate file. Default: "%s". Also available: "%s"' % (N3D, PDB))

    arg_parse.add_argument(
        '-split_chromosome',
        default=False,
        action='store_true',
        help='Split the output by chromosomes.')

    arg_parse.add_argument(
        '-s', nargs='+', default=[8.0,4.0,2.0,0.4,0.2,0.1], metavar='Mb_SIZE', type=float,
        help='One or more sizes (Mb) for the hierarchical structure calculation protocol (will be used in descending order). Default: 8.0 4.0 2.0 0.4 0.2 0.1')

    arg_parse.add_argument(
        '-cpu', metavar='NUM_CPU', type=int, default=MAX_CORES,
        help='Number of parallel CPU cores for calculating different coordinate models. Limited by the number of models (-m) but otherwise defaults to all available CPU cores (%d)' % MAX_CORES)

    arg_parse.add_argument(
        '-iso', default=2.0, metavar='Mb_SIZE', type=float,
        help='Contacts must be near another, within this (Mb) separation threshold (at both ends) to be considered supported: Default 2.0')

    arg_parse.add_argument(
        '-pow', default=-0.33, metavar='FLOAT',
        type=float, help='Distance power law for combining multiple Hi-C contacts between the same particles. Default: -0.33')

    arg_parse.add_argument(
        '-lower', default=0.8, metavar='DISTANCE',
        type=float, help='Lower limit for a contact-derived distance restraint, as a fraction of the ideal distance. Default: 0.8')

    arg_parse.add_argument(
        '-upper', default=1.2, metavar='DISTANCE',
        type=float, help='Upper limit for a contact-derived distance restraint, as a fraction of the ideal distance. Default: 1.2')

    arg_parse.add_argument(
        '-bb_lower', default=0.1, metavar='DISTANCE',
        type=float, help='Lower limit for sequential particle backbone restraints, as a fraction of the ideal distance. Default: 0.1')

    arg_parse.add_argument(
        '-bb_upper', default=1.1, metavar='DISTANCE',
        type=float, help='Upper limit for sequential particle backbone restraints, as a fraction of the ideal distance. Default: 1.1')

    arg_parse.add_argument(
        '-ran', metavar='INT',
        type=int, help='Seed for psuedo-random number generator')

    arg_parse.add_argument(
        '-rad', default=10.0, metavar='DISTANCE',
        type=float, help='Radius of sphere for random starting coordinates. Default: 10.0')

    arg_parse.add_argument(
        '-hot', default=5000.0, metavar='TEMP_KELVIN',
        type=float, help='Start annealing temperature in pseudo-Kelvin units. Default: 5000')

    arg_parse.add_argument(
        '-cold', default=10.0, metavar='TEMP_KELVIN',
        type=float, help='End annealing temperature in pseudo-Kelvin units. Default: 10')

    arg_parse.add_argument(
        '-temps', default=500, metavar='NUM_STEPS',
        type=int, help='Number of temperature steps in annealing protocol between start and end temperature. Default: 500')

    arg_parse.add_argument(
        '-dyns', default=100, metavar='NUM_STEPS',
        type=int, help='Number of particle dynamics steps to apply at each temperature in the annealing protocol. Default: 100')

    arg_parse.add_argument(
        '-time_step', default=0.001, metavar='TIME_DELTA',
        type=float, help='Simulation time step between re-calculation of particle velocities. Default: 0.001')
    
    args = vars(arg_parse.parse_args())
    
    input_file_path = args['input_file'][0]
    
    save_path = args['o']
    if save_path is None:
        save_path = os.path.splitext(input_file_path)[0]
        
    particle_sizes = args['s']
    particle_sizes = sorted([x * 1e6 for x in particle_sizes if x > 0], reverse=True)
    if not particle_sizes:
        critical('No positive particle sizes (Mb) specified')
    
    num_models = args['m']
    out_format = args['f'].lower()
    split_chromosome = args['split_chromosome']
    num_cpu = args['cpu'] or 1
    dist_power_law = args['pow']
    contact_dist_lower = args['lower']
    contact_dist_upper = args['upper']
    backbone_dist_lower = args['bb_lower']
    backbone_dist_upper = args['bb_upper']
    random_radius = args['rad']
    random_seed = args['ran']
    temp_start = args['hot']
    temp_end = args['cold']
    temp_steps = args['temps']
    dynamics_steps = args['dyns']
    time_step = args['time_step']
    isolation_threshold = args['iso']
    save_intermediate = args['save_intermediate']
    start_coords_path = args['start_coords_path']
    
    if out_format not in FORMATS:
        critical('Output file format must be one of: %s' % ', '.join(FORMATS))
    
    for val, name, sign in (
        (num_models, 'Number of conformational models', '+'),
        (dist_power_law, 'Distance power law', '-0'),
        (contact_dist_lower, 'Contact distance lower bound', '+'),
        (contact_dist_upper, 'Contact distance upper bound', '+'),
        (backbone_dist_lower, 'Backbone distance lower bound', '+'),
        (backbone_dist_upper, 'Backbone distance upper bound', '+'),
        (random_radius, 'Random-start radius', '+'),
        (temp_start, 'Annealing start temperature', '+'),
        (temp_end, 'Annealing end temperature', '+0'),
        (temp_steps, 'Number of annealing temperature steps', '+'),
        (dynamics_steps, 'Number of particle dynamics steps', '+'),
        (time_step, 'Particle dynamics time steps', '+'),
        (isolation_threshold, 'Contact isolation threshold', '+0')):

        if '+' in sign:
            if '0' in sign:
                if val < 0.0:
                    critical('%s must be non-negative' % name)
            else:
                if val <= 0.0:
                    critical('%s must be positive' % name)
        elif '-' in sign:    
            if '0' in sign:
                if val > 0.0:
                    critical('%s must be non-positive' % name)
            else:
                if val >= 0.0:
                    critical('%s must be negative' % name)
         
    contact_dist_lower, contact_dist_upper = sorted([contact_dist_lower, contact_dist_upper])
    backbone_dist_lower, backbone_dist_upper = sorted([backbone_dist_lower, backbone_dist_upper])
    temp_end, temp_start = sorted([temp_end, temp_start])
    
    if not random_seed:
        random_seed = int(time())
        
    general_calc_params = {
        'dist_power_law':dist_power_law,
        'contact_dist_lower':contact_dist_lower,
        'contact_dist_upper':contact_dist_upper,
        'backbone_dist_lower':backbone_dist_lower,
        'backbone_dist_upper':backbone_dist_upper,
        'random_seed':random_seed,
        'random_radius':random_radius
    }

    anneal_params = {
        'temp_start':temp_start, 'temp_end':temp_end, 'temp_steps':temp_steps,
        'dynamics_steps':dynamics_steps, 'time_step':time_step
    }

    isolation_threshold *= 1e6

    calc_genome_structure(input_file_path, save_path, general_calc_params, anneal_params,
                          particle_sizes, num_models, isolation_threshold, out_format, split_chromosome,
                          num_cpu, start_coords_path, save_intermediate)

