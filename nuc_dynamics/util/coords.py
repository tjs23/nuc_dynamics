import numpy as np


def get_random_coords(pos_dict, chromosomes, num_models, radius=10.0):
    """
    Get random, uniformly sampled coorinate positions, restricted to
    a sphere of given radius
    """
    from numpy.random import uniform

    num_particles = sum([len(pos_dict[chromo]) for chromo in chromosomes])
    coords = np.empty((num_models, num_particles, 3))
    r2 = radius*radius

    for m in range(num_models):
        for i in range(num_particles):
            x = y = z = radius

            while x*x + y*y + z*z >= r2:
                x = radius * (2*uniform(0,1) - 1)
                y = radius * (2*uniform(0,1) - 1)
                z = radius * (2*uniform(0,1) - 1)

            coords[m,i] = [x,y,z]

    return coords


def pack_chromo_coords(coords_dict, chromosomes):
    """
    Place chromosome 3D coordinates stored in a dictionary keyed by
    chromosome name into a single, ordered array. The chromosomes argument
    is required to set the correct array storage order.
    """

    chromo_num_particles = [len(coords_dict[chromo][0]) for chromo in chromosomes]
    n_particles = sum(chromo_num_particles)
    n_models = len(coords_dict[chromosomes[0]])    
    coords = np.empty((n_models, n_particles, 3), float)
    
    j = 0
    for i, chromo in enumerate(chromosomes):
        span = chromo_num_particles[i]
        coords[:,j:j+span] = coords_dict[chromo]
        j += span
            
    return coords

 
def unpack_chromo_coords(coords, chromosomes, seq_pos_dict):
    """
    Exctract coords for multiple chromosomes stored in a single array into
    a dictionary, keyed by chromosome name. The chromosomes argument is required
    to get the correct array storage order.
    """

    chromo_num_particles = [len(seq_pos_dict[chromo]) for chromo in chromosomes]
    n_seq_pos = sum(chromo_num_particles)
    n_models, n_particles, dims = coords.shape

    if n_seq_pos != n_particles:
        msg = 'Model coordinates must be an array of num models x %d' % (n_seq_pos,)
        raise(Exception(msg))    
    
    coords_dict = {}
                
    j = 0
    for i, chromo in enumerate(chromosomes):
        span = chromo_num_particles[i]
        coords_dict[chromo] = coords[:,j:j+span] # all models, slice
        j += span
 
    return coords_dict