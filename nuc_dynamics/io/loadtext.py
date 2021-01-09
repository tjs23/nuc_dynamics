import os
import gzip
import numpy as np


def open_file(file_path, mode=None, gzip_exts=('.gz','.gzip')):
    """
    GZIP agnostic file opening
    """
    IO_BUFFER = int(4e6)
    
    if os.path.splitext(file_path)[1].lower() in gzip_exts:
        file_obj = gzip.open(file_path, mode or 'rt')
    else:
        file_obj = open(file_path, mode or 'rU', IO_BUFFER)
 
    return file_obj


def load_pairs_file(file_path):
    """
    Load 4DCIC pairs file

    Return
    ------
    chromosomes : List[str]
    chromo : Dict[str, List[int, int]]
    contact_dict : Dict[str, Dict[str, List[int, int, int, int]]]
        chr_a -> (chr_b -> (pos1, pos2, num_obs, ambig_group))
    """
    file_obj = open_file(file_path)

    contact_dict = {}
    chromosomes = set()

    num_obs = 1
    ambig_group = 0

    for line in file_obj:
        if line.startswith("#"): continue
        read_id, chr_a, f_start_a, chr_b, f_start_b, strand_a, strand_b = line.split()
        pos_a = int(f_start_a)
        pos_b = int(f_start_b)

        if chr_a > chr_b:
            chr_a, chr_b = chr_b, chr_a
            pos_a, pos_b = pos_b, pos_a

        if chr_a not in contact_dict:
            contact_dict[chr_a] = {}
            chromosomes.add(chr_a)

        if chr_b not in contact_dict[chr_a]:
            contact_dict[chr_a][chr_b] = [] 
            chromosomes.add(chr_b)
                
        contact_dict[chr_a][chr_b].append((pos_a, pos_b, num_obs, ambig_group))

    file_obj.close()

    chromo_limits = {}

    for chr_a in contact_dict:
        for chr_b in contact_dict[chr_a]:
            contacts = np.array(contact_dict[chr_a][chr_b]).T
            contact_dict[chr_a][chr_b] = contacts
            
            seq_pos_a = contacts[0]
            seq_pos_b = contacts[1]
            
            min_a = min(seq_pos_a)
            max_a = max(seq_pos_a)
            min_b = min(seq_pos_b)
            max_b = max(seq_pos_b)
                
            if chr_a in chromo_limits:
                prev_min, prev_max = chromo_limits[chr_a]
                chromo_limits[chr_a] = [min(prev_min, min_a), max(prev_max, max_a)]
            else:
                chromo_limits[chr_a] = [min_a, max_a]
            
            if chr_b in chromo_limits:
                prev_min, prev_max = chromo_limits[chr_b]
                chromo_limits[chr_b] = [min(prev_min, min_b), max(prev_max, max_b)]
            else:
                chromo_limits[chr_b] = [min_b, max_b]

    chromosomes = sorted(chromosomes)            

    return chromosomes, chromo_limits, contact_dict


def load_ncc_file(file_path):
    """Load chromosome and contact data from NCC format file, as output from NucProcess"""
    
    if file_path.endswith('.gz'):
        import gzip
        file_obj = gzip.open(file_path)
    
    else:
        file_obj = open(file_path) 
    
    # Observations are treated individually in single-cell Hi-C,
    # i.e. no binning, so num_obs always 1 for each contact
    num_obs = 1    
        
    contact_dict = {}
    chromosomes = set()
    ambig_group = 0
        
    for line in file_obj:
        chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, ambig_group, pair_id, swap_pair = line.split()
        
        if '.' in ambig_group: # Updated NCC format
            if int(float(ambig_group)) > 0:
                ambig_group += 1
        else:
            ambig_group = int(ambig_group) 
                    
        if strand_a == '+':
            pos_a = int(f_start_a)
        else:
            pos_a = int(f_end_a)
        
        if strand_b == '+':
            pos_b = int(f_start_b)             
        else:
            pos_b = int(f_end_b)
 
        if chr_a > chr_b:
            chr_a, chr_b = chr_b, chr_a
            pos_a, pos_b = pos_b, pos_a
        
        if chr_a not in contact_dict:
            contact_dict[chr_a] = {}
            chromosomes.add(chr_a)
            
        if chr_b not in contact_dict[chr_a]:
            contact_dict[chr_a][chr_b] = [] 
            chromosomes.add(chr_b)
                
        contact_dict[chr_a][chr_b].append((pos_a, pos_b, num_obs, ambig_group))

    file_obj.close()

    chromo_limits = {}

    for chr_a in contact_dict:
        for chr_b in contact_dict[chr_a]:
            contacts = np.array(contact_dict[chr_a][chr_b]).T
            contact_dict[chr_a][chr_b] = contacts
            
            seq_pos_a = contacts[0]
            seq_pos_b = contacts[1]
            
            min_a = min(seq_pos_a)
            max_a = max(seq_pos_a)
            min_b = min(seq_pos_b)
            max_b = max(seq_pos_b)
                
            if chr_a in chromo_limits:
                prev_min, prev_max = chromo_limits[chr_a]
                chromo_limits[chr_a] = [min(prev_min, min_a), max(prev_max, max_a)]
            else:
                chromo_limits[chr_a] = [min_a, max_a]
            
            if chr_b in chromo_limits:
                prev_min, prev_max = chromo_limits[chr_b]
                chromo_limits[chr_b] = [min(prev_min, min_b), max(prev_max, max_b)]
            else:
                chromo_limits[chr_b] = [min_b, max_b]
                 
    chromosomes = sorted(chromosomes)            
                
    return chromosomes, chromo_limits, contact_dict


def load_n3d_coords(file_path):
    """
    Load genome structure coordinates and particle sequence positions from an N3D format file.

    Args:
            file_path: str ; Location of N3D (text) format file

    Returns:
            dict {str:ndarray(n_coords, int)}                                    ; {chromo: seq_pos_array}
            dict {str:ndarray((n_models, n_coords, 3), float)} ; {chromo: coord_3d_array}

    """
    seq_pos_dict = {}
    coords_dict = {}

    with open_file(file_path) as file_obj:
       chromo = None

       for line in file_obj:

           data = line.split()
           n_items = len(data)

           if not n_items:
               continue

           elif data[0] == '#':
               continue

           elif n_items == 3:
               chromo, n_coords, n_models = data

               #if chromo.lower()[:3] == 'chr':
               #    chromo = chromo[3:]

               n_coords = int(n_coords)
               n_models = int(n_models)

               #chromo_seq_pos = np.empty(n_coords, int)
               chromo_seq_pos = np.empty(n_coords, 'int32')
               chromo_coords = np.empty((n_models, n_coords, 3), float)

               coords_dict[chromo]    = chromo_coords
               seq_pos_dict[chromo] = chromo_seq_pos

               check = (n_models * 3) + 1
               i = 0

           elif not chromo:
               raise Exception('Missing chromosome record in file %s' % file_path)

           elif n_items != check:
               msg = 'Data size in file %s does not match Position + Models * Positions * 3'
               raise Exception(msg % file_path)

           else:
               chromo_seq_pos[i] = int(data[0])

               coord = [float(x) for x in data[1:]]
               coord = np.array(coord).reshape(n_models, 3)
               chromo_coords[:,i] = coord
               i += 1

    return seq_pos_dict, coords_dict
