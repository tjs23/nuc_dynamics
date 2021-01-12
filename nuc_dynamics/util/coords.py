import numpy as np


def get_random_coords(pos_dict, chromosomes, num_models, radius=10.0):
    """
    Get random, uniformly sampled coorinate positions, restricted to
    a sphere of given radius

    Return
    ------
    coords : ndarray[double, dim=3]
        In shape (num_models, num_particles, 3).
    """
    from numpy.random import uniform

    num_particles = sum([len(pos_dict[chromo]) for chromo in chromosomes])
    coords = np.empty((num_models, num_particles, 3))
    for m in range(num_models):
        r = uniform(0, radius, size=num_particles)
        theta = uniform(0, 2*np.pi, size=num_particles)
        phi = uniform(0, np.pi, size=num_particles)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)
        pos = np.c_[x, y, z]
        coords[m] = pos

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
