# Python functions for NucDynamics
import sys
import numpy as np
import multiprocessing
import traceback

PROG_NAME = 'nuc_dynamics'
DESCRIPTION = 'Single-cell Hi-C genome and chromosome structure calculation module for Nuc3D and NucTools'

N3D = 'n3d'
PDB = 'pdb'
FORMATS = [N3D, PDB]
MAX_CORES = multiprocessing.cpu_count()

def warn(msg, prefix='WARNING'):

  print('%8s : %s' % (prefix, msg))


def critical(msg, prefix='ABORT'):

  print('%8s : %s' % (prefix, msg))
  sys.exit(0)
  
  
def _parallel_func_wrapper(queue, target_func, proc_data, common_args):
  
  for t, data_item in proc_data:
    result = target_func(data_item, *common_args)
    
    if queue:
      queue.put((t, result))
    
    elif isinstance(result, Exception):
      raise(result)


def parallel_split_job(target_func, split_data, common_args, num_cpu=MAX_CORES, collect_output=True):
  
  num_tasks   = len(split_data)
  num_process = min(num_cpu, num_tasks)
  
  processes   = []
  
  if collect_output:
    queue = multiprocessing.Queue() # Queue will collect parallel process output
  
  else:
    queue = None
    
  for p in range(num_process):
    # Task IDs and data for each task
    # Each process can have multiple tasks if there are more tasks than processes/cpus
    proc_data = [(t, data_item) for t, data_item in enumerate(split_data) if t % num_cpu == p]
    args = (queue, target_func, proc_data, common_args)

    proc = multiprocessing.Process(target=_parallel_func_wrapper, args=args)
    processes.append(proc)
  
  for proc in processes:
    proc.start()
  
  if queue:
    results = [None] * num_tasks
    
    for i in range(num_tasks):
      t, result = queue.get() # Asynchronous fetch output: whichever process completes a task first
      
      if isinstance(result, Exception):
        print('\n* * * * C/Cython code may need to be recompiled. Try running "python setup_cython.py build_ext --inplace" * * * *\n')
        raise(result)
        
      results[t] = result
 
    queue.close()
 
    return results
  
  else:
    for proc in processes: # Asynchromous wait and no output captured
      proc.join()


def load_pairs_file(file_path):
  """Load 4DCIC pairs file"""
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


def export_n3d_coords(file_path, coords_dict, seq_pos_dict):
  
  file_obj = open(file_path, 'w')
  write = file_obj.write
  
  for chromo in seq_pos_dict:
    chromo_coords = coords_dict[chromo]
    chromo_seq_pos = seq_pos_dict[chromo]
    
    num_models = len(chromo_coords)
    num_coords = len(chromo_seq_pos)
    
    line = '%s\t%d\t%d\n' % (chromo, num_coords, num_models)
    write(line)
    
    for j in range(num_coords):
      data = chromo_coords[:,j].ravel().tolist()
      data = '\t'.join('%.8f' % d for d in  data)
      
      line = '%d\t%s\n' % (chromo_seq_pos[j], data)
      write(line)

  file_obj.close()


def export_pdb_coords(file_path, coords_dict, seq_pos_dict, particle_size, scale=1.0, extended=True):
  """
  Write chromosome particle coordinates as a PDB format file
  """

  alc = ' '
  ins = ' '
  prefix = 'HETATM'
  line_format = '%-80.80s\n'
  
  if extended:
    pdb_format = '%-6.6s%5.1d %4.4s%s%3.3s %s%4.1d%s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2.2s  %10d\n'
    ter_format = '%-6.6s%5.1d      %s %s%4.1d%s                                                     %10d\n'
  else:
    pdb_format = '%-6.6s%5.1d %4.4s%s%3.3s %s%4.1d%s   %8.3f%8.3f%8.3f%6.2f%6.2f          %2.2s  \n'
    ter_format = '%-6.6s%5.1d      %s %s%4.1d%s                                                     \n'

  file_obj = open(file_path, 'w')
  write = file_obj.write
  
  chromosomes = list(seq_pos_dict.keys())
  
  sort_chromos = []
  for chromo in chromosomes:
    if chromo[:3].lower() == 'chr':
      key = chromo[3:]
    else:
      key = chromo
    
    if key.isdigit():
      key = '%03d' % int(key)
    
    sort_chromos.append((key, chromo))
  
  sort_chromos.sort()
  sort_chromos = [x[1] for x in sort_chromos]
  
  num_models = len(coords_dict[chromosomes[0]])
  title = 'NucDynamics genome structure export'
  
  write(line_format % 'TITLE     %s' % title)
  write(line_format % 'REMARK 210') 
  write(line_format % 'REMARK 210 Atom type C is used for all particles')
  write(line_format % 'REMARK 210 Atom number increases every %s bases' % particle_size)
  write(line_format % 'REMARK 210 Residue code indicates chromosome')
  write(line_format % 'REMARK 210 Residue number represents which sequence Mb the atom is in')
  write(line_format % 'REMARK 210 Chain letter is different every chromosome, where Chr1=a, Chr2=b etc.')
  
  if extended:
    file_obj.write(line_format % 'REMARK 210 Extended PDB format with particle seq. pos. in last column')
  
  file_obj.write(line_format % 'REMARK 210')
  
  pos_chromo = {}
  
  for m in range(num_models):
    line = 'MODEL     %4d' % (m+1)
    write(line_format  % line)
    
    c = 0
    j = 1
    seqPrev = None
    
    for k, chromo in enumerate(sort_chromos):
      chain_code = chr(ord('A')+k)            
      
      tlc = chromo
      if tlc.lower().startswith("chr"):
        tlc = tlc[3:]
      while len(tlc) < 2:
        tlc = '_' + tlc
      
      if len(tlc) == 2:
        tlc = 'C' + tlc
      
      if len(tlc) > 3:
        tlc = tlc[:3]
      
      chromo_model_coords = coords_dict[chromo][m]
      
      if not len(chromo_model_coords):
        continue
      
      pos = seq_pos_dict[chromo]
      
      for i, seqPos in enumerate(pos):
        c += 1
 
        seqMb = int(seqPos//1e6) + 1
        
        if seqMb == seqPrev:
          j += 1
        else:
          j = 1
        
        el = 'C'
        a = 'C%d' % j
          
        aName = '%-3s' % a
        x, y, z = chromo_model_coords[i] #XYZ coordinates
         
        seqPrev = seqMb
        pos_chromo[c] = chromo
        
        if extended:
          line  = pdb_format % (prefix,c,aName,alc,tlc,chain_code,seqMb,ins,x,y,z,0.0,0.0,el,seqPos)
        else:
          line  = pdb_format % (prefix,c,aName,alc,tlc,chain_code,seqMb,ins,x,y,z,0.0,0.0,el)
          
        write(line)
 
    write(line_format  % 'ENDMDL')
 
  for i in range(c-2):
     if pos_chromo[i+1] == pos_chromo[i+2]:
       line = 'CONECT%5.1d%5.1d' % (i+1, i+2)
       write(line_format  % line)
 
  write(line_format  % 'END')
  file_obj.close()


def remove_isolated_contacts(contact_dict, threshold=int(2e6)):
  """
  Select only contacts which are within a given sequence separation of another
  contact, for the same chromosome pair
  """
    
  for chromoA in contact_dict:
    for chromoB in contact_dict[chromoA]:
      contacts = contact_dict[chromoA][chromoB]
      positions = np.array(contacts[:2], np.int32).T
      
      if len(positions): # Sometimes empty e.g. for MT, Y chromos 
        active_idx = dyn_util.getSupportedPairs(positions, np.int32(threshold))
        contact_dict[chromoA][chromoB] = contacts[:,active_idx]
     
  return contact_dict
  

def remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos, particle_size, threshold=5.0):  
  """
  Remove contacts whith structure distances that exceed a given threshold
  """
  
  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contacts = contact_dict[chr_a][chr_b]
      
      contact_pos_a = contacts[0].astype(np.int32)
      contact_pos_b = contacts[1].astype(np.int32)
      
      coords_a = coords_dict[chr_a]
      coords_b = coords_dict[chr_b]

      struc_dists = []
 
      for m in range(len(coords_a)):
        coord_data_a = dyn_util.getInterpolatedCoords([chr_a], {chr_a:contact_pos_a}, particle_seq_pos, coords_a[m])
        coord_data_b = dyn_util.getInterpolatedCoords([chr_b], {chr_b:contact_pos_b}, particle_seq_pos, coords_b[m])
 
        deltas = coord_data_a - coord_data_b
        dists = np.sqrt((deltas*deltas).sum(axis=1))
        struc_dists.append(dists)
      
      # Average over all conformational models
      struc_dists = np.array(struc_dists).T.mean(axis=1)
      
      # Select contacts with distances below distance threshold
      indices = (struc_dists < threshold).nonzero()[0]
      contact_dict[chr_a][chr_b] = contacts[:,indices]
        
  return contact_dict
  
  
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
  
  print('  starting model %d' % m)
  
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

  print('  done model %d' % m)
  
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
    restraint_indices, restraint_dists = dyn_util.concatenate_restraints(restraint_dict, seq_pos_dict, particle_size,
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
    masses = np.ones(num_coords,  float)
    radii = np.ones(num_coords,  float)
    
    # Ambiguiity strides not used here, so set to 1
    num_restraints = len(restraint_indices)
    ambiguity = np.ones(num_restraints,  np.int32)
        
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

def load_n3d_coords(file_path):
 """
 Load genome structure coordinates and particle sequence positions from an N3D format file.

 Args:
     file_path: str ; Location of N3D (text) format file

 Returns:
     dict {str:ndarray(n_coords, int)}                  ; {chromo: seq_pos_array}
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
       #  chromo = chromo[3:]

       n_coords = int(n_coords)
       n_models = int(n_models)

       #chromo_seq_pos = np.empty(n_coords, int)
       chromo_seq_pos = np.empty(n_coords, 'int32')
       chromo_coords = np.empty((n_models, n_coords, 3), float)

       coords_dict[chromo]  = chromo_coords
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


def export_coords(out_format, out_file_path, coords_dict, particle_seq_pos, particle_size):
  
  # Save final coords as N3D or PDB format file
  
  if out_format == PDB:
    if not out_file_path.endswith(PDB):
      out_file_path = '%s.%s' % (out_file_path, PDB)
  
    export_pdb_coords(out_file_path, coords_dict, particle_seq_pos, particle_size)
 
  else:
    if not out_file_path.endswith(N3D):
      out_file_path = '%s.%s' % (out_file_path, N3D)
      
    export_n3d_coords(out_file_path, coords_dict, particle_seq_pos)
    
  print('Saved structure file to: %s' % out_file_path)


def calc_genome_structure(input_file_path, out_file_path, general_calc_params, anneal_params,
                          particle_sizes, num_models=5, isolation_threshold=2e6,
                          out_format=N3D, num_cpu=MAX_CORES,
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
  export_coords(out_format, out_file_path, coords_dict, particle_seq_pos, particle_size)


def test_imports(gui=False):
  import sys
  from distutils.core import run_setup
  
  critical = False
  
  try:
    import numpy
  except ImportError as err:
    critical = True
    warn('Critical Python module "numpy" is not installed or accessible')

  try:
    import cython
  except ImportError as err:
    critical = True
    warn('Critical Python module "cython" is not installed or accessible')
  
  try:
    import dyn_util
  except ImportError as err:
    import os
    cwd = os.getcwd()
    try:
      os.chdir(os.path.dirname(os.path.normpath(__file__)))
      warn('Utility C/Cython code not compiled. Attempting to compile now...')    
      run_setup('setup_cython.py', ['build_ext', '--inplace'])
    finally:
      os.chdir(cwd)
    
    try:
      import dyn_util
      warn('NucDynamics C/Cython code compiled. Please re-run command.')
      sys.exit(0)
      
    except ImportError as err:
      critical = True
      warn('Utility C/Cython code compilation/import failed')   
    
  if critical:
    warn('NucDynamics cannot proceed because critical Python modules are not available', 'ABORT')
    sys.exit(0)
    
    
def demo_calc_genome_structure():
  """
  Example of settings for a typical genome structure calculation from input single-cell
  Hi-C contacts stored in an NCC format file (as output from NucProcess)
  """
  
  from nuc_dynamics import calc_genome_structure
  
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

import dyn_util

if __name__ == '__main__':
  
  import os, sys
  from time import time
  from argparse import ArgumentParser
  
  epilog = 'For further help on running this program please email tjs23@cam.ac.uk'
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument('input_file', nargs=1, metavar='INPUT_FILE',
                         help='Input NCC/PAIRs format file containing single-cell Hi-C contact data, e.g. use the demo data at example_chromo_data/Cell_1_contacts.ncc')

  arg_parse.add_argument('-o', metavar='OUT_FILE',
                         help='Optional name of output file for 3D coordinates in N3D or PDB format (see -f option). If not set this will be auto-generated from the input file name')

  arg_parse.add_argument('-save_intermediate', default=False, action='store_true',
                         help='Write out intermediate coordinate files.')

  arg_parse.add_argument('-start_coords_path', metavar='N3D_FILE',
                         help='Initial 3D coordinates in N3D format. If set this will override -m flag.')

  arg_parse.add_argument('-m', default=1, metavar='NUM_MODELS',
                         type=int, help='Number of alternative conformations to generate from repeat calculations with different random starting coordinates: Default: 1')

  arg_parse.add_argument('-f', metavar='OUT_FORMAT', default=N3D,
                         help='File format for output 3D coordinate file. Default: "%s". Also available: "%s"' % (N3D, PDB))

  arg_parse.add_argument('-s', nargs='+', default=[8.0,4.0,2.0,0.4,0.2,0.1], metavar='Mb_SIZE', type=float,
                         help='One or more sizes (Mb) for the hierarchical structure calculation protocol (will be used in descending order). Default: 8.0 4.0 2.0 0.4 0.2 0.1')

  arg_parse.add_argument('-cpu', metavar='NUM_CPU', type=int, default=MAX_CORES,
                         help='Number of parallel CPU cores for calculating different coordinate models. Limited by the number of models (-m) but otherwise defaults to all available CPU cores (%d)' % MAX_CORES)

  arg_parse.add_argument('-iso', default=2.0, metavar='Mb_SIZE', type=float,
                         help='Contacts must be near another, within this (Mb) separation threshold (at both ends) to be considered supported: Default 2.0')

  arg_parse.add_argument('-pow', default=-0.33, metavar='FLOAT',
                         type=float, help='Distance power law for combining multiple Hi-C contacts between the same particles. Default: -0.33')

  arg_parse.add_argument('-lower', default=0.8, metavar='DISTANCE',
                         type=float, help='Lower limit for a contact-derived distance restraint, as a fraction of the ideal distance. Default: 0.8')

  arg_parse.add_argument('-upper', default=1.2, metavar='DISTANCE',
                         type=float, help='Upper limit for a contact-derived distance restraint, as a fraction of the ideal distance. Default: 1.2')

  arg_parse.add_argument('-bb_lower', default=0.1, metavar='DISTANCE',
                         type=float, help='Lower limit for sequential particle backbone restraints, as a fraction of the ideal distance. Default: 0.1')

  arg_parse.add_argument('-bb_upper', default=1.1, metavar='DISTANCE',
                         type=float, help='Upper limit for sequential particle backbone restraints, as a fraction of the ideal distance. Default: 1.1')

  arg_parse.add_argument('-ran', metavar='INT',
                         type=int, help='Seed for psuedo-random number generator')

  arg_parse.add_argument('-rad', default=10.0, metavar='DISTANCE',
                         type=float, help='Radius of sphere for random starting coordinates. Default: 10.0')

  arg_parse.add_argument('-hot', default=5000.0, metavar='TEMP_KELVIN',
                         type=float, help='Start annealing temperature in pseudo-Kelvin units. Default: 5000')

  arg_parse.add_argument('-cold', default=10.0, metavar='TEMP_KELVIN',
                         type=float, help='End annealing temperature in pseudo-Kelvin units. Default: 10')

  arg_parse.add_argument('-temps', default=500, metavar='NUM_STEPS',
                         type=int, help='Number of temperature steps in annealing protocol between start and end temperature. Default: 500')

  arg_parse.add_argument('-dyns', default=100, metavar='NUM_STEPS',
                         type=int, help='Number of particle dynamics steps to apply at each temperature in the annealing protocol. Default: 100')

  arg_parse.add_argument('-time_step', default=0.001, metavar='TIME_DELTA',
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
  num_cpu    = args['cpu'] or 1
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
  
  for val, name, sign in ((num_models,         'Number of conformational models', '+'),
                          (dist_power_law,     'Distance power law', '-0'),
                          (contact_dist_lower, 'Contact distance lower bound', '+'),
                          (contact_dist_upper, 'Contact distance upper bound', '+'),
                          (backbone_dist_lower,'Backbone distance lower bound', '+'),
                          (backbone_dist_upper,'Backbone distance upper bound', '+'),
                          (random_radius,      'Random-start radius', '+'),
                          (temp_start,         'Annealing start temperature', '+'),
                          (temp_end,           'Annealing end temperature', '+0'),
                          (temp_steps,         'Number of annealing temperature steps', '+'),
                          (dynamics_steps,     'Number of particle dynamics steps', '+'),
                          (time_step,          'Particle dynamics time steps', '+'),
                          (isolation_threshold,'Contact isolation threshold', '+0'),
                         ):
                          
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
    
  general_calc_params = {'dist_power_law':dist_power_law,
                         'contact_dist_lower':contact_dist_lower,
                         'contact_dist_upper':contact_dist_upper,
                         'backbone_dist_lower':backbone_dist_lower,
                         'backbone_dist_upper':backbone_dist_upper,
                         'random_seed':random_seed,
                         'random_radius':random_radius}

  anneal_params = {'temp_start':temp_start, 'temp_end':temp_end, 'temp_steps':temp_steps,
                   'dynamics_steps':dynamics_steps, 'time_step':time_step}
  
  isolation_threshold *= 1e6
  
  calc_genome_structure(input_file_path, save_path, general_calc_params, anneal_params,
                        particle_sizes, num_models, isolation_threshold, out_format, num_cpu,
                        start_coords_path, save_intermediate)

# TO DO
# -----
# Allow chromosomes to be specified
# Allow starting structures to be input


