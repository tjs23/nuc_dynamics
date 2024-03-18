# Python functions for NucDynamics
import os
import sys
import numpy as np
import multiprocessing
import subprocess
import traceback
import gzip

QUIET            = False # Global verbosity flag
LOG_FILE_OBJ     = None # Created if needed

PROG_NAME = 'nuc_dynamics'
VERSION = '1.3.0'
DESCRIPTION = 'Single-cell Hi-C genome and chromosome structure calculation module for Nuc3D and NucTools'

FILE_BUFFER_SIZE = 2**16
N3D = 'n3d'
PDB = 'pdb'
FORMATS = [N3D, PDB]
MAX_CORES = multiprocessing.cpu_count()

DEFAULT_STRUCT_AMBIG_SIZE_MB = 0.4
DEFAULT_REMOVE_MODELS_SIZE_MB = 1

DEFAULT_STRUCT_AMBIG_SIZE = DEFAULT_STRUCT_AMBIG_SIZE_MB * int(1e6)
DEFAULT_REMOVE_MODELS_SIZE = DEFAULT_REMOVE_MODELS_SIZE_MB * int(1e6)

Restraint = np.dtype([('indices', 'int32', 2), ('dists', 'float64', 2),
                      ('ambiguity', 'int32'),  ('weight', 'float64')])
Contact = np.dtype([('pos', 'uint32', 2), ('ambiguity', 'int32'), ('number', 'uint32')])

def report(msg, prefix=''):

  if prefix:
    msg = '%s: %s' % (prefix, msg)
    
  if LOG_FILE_OBJ:
    LOG_FILE_OBJ.write(msg + '\n')
    LOG_FILE_OBJ.flush()
    if not QUIET:
      print(msg)
  else:
    print(msg)


def info(msg, prefix='INFO'):

  report(msg, prefix)


def warn(msg, prefix='WARNING'):

  report(msg, prefix)


def critical(msg, prefix='ABORT'):

  report(msg, prefix)
  sys.exit(0)

  
def _parallel_func_wrapper(queue, target_func, proc_data, common_args):
  
  for t, data_item in proc_data:
    result = target_func(data_item, *common_args)
    
    if queue:
      queue.put((t, result))
    
    elif isinstance(result, Exception):
      raise(result)


def open_file(file_path, mode=None, buffer_size=FILE_BUFFER_SIZE, gzip_exts=('.gz','.gzip'), partial=False):
  """
  GZIP agnostic file opening
  """
  import io
  
  if os.path.splitext(file_path)[1].lower() in gzip_exts:
    if mode and 'w' in mode:
      file_obj = io.BufferedWriter(gzip.open(file_path, mode), buffer_size)
      
    else:
      if partial:
        file_obj = io.BufferedReader(gzip.open(file_path, mode or 'rb'), buffer_size)
        
      else:
        try:
          file_obj = subprocess.Popen(['zcat', file_path], stdout=subprocess.PIPE).stdout
        except OSError:
          file_obj = io.BufferedReader(gzip.open(file_path, mode or 'rb'), buffer_size)
    
    if sys.version_info.major > 2:
      file_obj = io.TextIOWrapper(file_obj, encoding="utf-8")
 
  else:
    if sys.version_info.major > 2:
      file_obj = open(file_path, mode or 'r', buffer_size, encoding='utf-8')
      
    else:
      file_obj = open(file_path, mode or 'r', buffer_size)
  
  return file_obj


def parallel_split_job(target_func, split_data, common_args, num_cpu=MAX_CORES, collect_output=True):
  
  num_tasks   = len(split_data)
  num_process = min(num_cpu, num_tasks)
  
  if num_process == 1:
    
    results = []
    for data_item in split_data:
      result = target_func(data_item, *common_args)
      
      if isinstance(result, Exception):
        raise(result)
        
      results.append(result)
    
    return results
  
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
        warn(result)
        critical('\n* * * * C/Cython code may need to be recompiled. Try running "python setup_cython.py build_ext --inplace" * * * *\n')
        
      results[t] = result
 
    queue.close()
 
    return results
  
  else:
    for proc in processes: # Asynchromous wait and no output captured
      proc.join()


def load_ncc_file(file_path, active_only=True, nmax=int(1e6)):
  """Load chromosome and contact data from NCC format file, as output from NucProcess"""

  from numpy import array

  contact_dict = {}
  ambig_group = 0

  with open_file(file_path) as file_obj:
    n_active = 0
    
    for i, line in enumerate(file_obj):
      if line.startswith('#'):
        continue
        
      row = line.split()
      if not '.' in row[12]:
        critical('The old version of the NCC format is not supported by the version of NucDynamics. Please use the latest version of NucProcess')
    
    ambig_group = 0
    file_obj.seek(0)   
    for i, line in enumerate(file_obj):
      if line.startswith('#'):
        continue
        
      chr_a, f_start_a, f_end_a, start_a, end_a, strand_a, chr_b, f_start_b, f_end_b, start_b, end_b, strand_b, ambig_code, pair_id, swap_pair = line.split()
      
      if int(float(ambig_code)) > 0:
        ambig_group += 1
      
      if active_only and ambig_code.endswith('.0'): # inactive
        continue
        
      pos_a = int(f_start_a if strand_a == '+' else f_end_a)
      pos_b = int(f_start_b if strand_b == '+' else f_end_b)

      if chr_a > chr_b:
        chr_a, chr_b = chr_b, chr_a
        pos_a, pos_b = pos_b, pos_a

      if chr_a not in contact_dict:
        contact_dict[chr_a] = {}

      if chr_b not in contact_dict[chr_a]:
        contact_dict[chr_a][chr_b] = []

      contact_dict[chr_a][chr_b].append(((pos_a, pos_b), ambig_group, i)) # Uses original file line number
      n_active += 1

  if n_active > nmax:
    critical('Too many contacts in ncc file (> {:,}), this code is meant for single-cell data'.format(nmax))
    
  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      contact_dict[chr_a][chr_b] = array(contact_dict[chr_a][chr_b], dtype=Contact)
      
  return contact_dict


def save_ncc_file(file_path_in, name, contact_dict, particle_size=None, offset=0):

  if particle_size is not None:
    name = '%s_%d' % (name, int(particle_size/1000))
  
  file_root, file_ext = os.path.splitext(file_path_in)
  
  if file_ext.lower() in ('.gz','.gzip'):
    file_root, file_ext = os.path.splitext(file_root)
 
  file_path_out ='%s_%s.ncc' % (file_root, name)
  
  active_numbers = set()
  for chr_a in contact_dict:
    for chr_b in contact_dict[chr_a]:
      active_numbers |= set(contact_dict[chr_a][chr_b]['number'])
      
  with open_file(file_path_in) as file_obj_in:
    with open_file(file_path_out, 'w') as file_obj_out:
     
      n = offset
      for i, line in enumerate(file_obj_in):
        if line.startswith('#'):
          file_obj_out.write(line)
        
        else:
          row = line.split()
          ag = row[12]  
          
          activ = '1' if i in active_numbers else '0'
          row[12] = ag[:-1] + activ            
          
          line = ' '.join(row) + '\n'  
          file_obj_out.write(line)

  
def calc_limits(contact_dict):
  chromo_limits = {}

  for chrs, contacts in flatten_dict(contact_dict).items():
    if len(contacts) < 1:
      continue

    for chromo, seq_pos in zip(chrs, contacts['pos'].T):
      min_pos = seq_pos.min()
      max_pos = seq_pos.max()

      prev_min, prev_max = chromo_limits.get(chromo, (min_pos, max_pos))
      chromo_limits[chromo] = [min(prev_min, min_pos), max(prev_max, max_pos)]
  return chromo_limits


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
    if chromo[:3] == 'chr':
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
      chain_code = chr(ord('a')+k)            
      
      tlc = chromo
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
        x *= scale
        y *= scale
        z *= scale
         
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


def remove_ambiguous_contacts(contact_dict):
  
  ambiguity_list = []
  for contacts in flatten_dict(contact_dict).values():
    ambiguity_list.extend(contacts['ambiguity'])
    
  ambiguity_array = np.array(sorted(ambiguity_list))
  
  _, inverse, counts = np.unique(
    ambiguity_array, return_inverse=True, return_counts=True
  )
  
  ambiguity_set = set(ambiguity_array[counts[inverse] == 1])

  from collections import defaultdict
  new_contacts = defaultdict(dict)

  for (chromoA, chromoB), contacts in flatten_dict(contact_dict).items():
    contacts = [c for c in contacts if c['ambiguity'] in ambiguity_set]
    if contacts:
      new_contacts[chromoA][chromoB] = np.array(contacts, dtype=Contact)
  
  return dict(new_contacts)
  

def homologous_pair(chromoA, chromoB):
  
  # slightly dangerous code, relies on chromoA/B being of form 'chr11.a', etc.
  nA = chromoA.rfind('.')
  nB = chromoB.rfind('.')
  
  if nA < 0 or nB < 0:
    return False
    
  return (chromoA[:nA] == chromoB[:nB]) and (chromoA[nA+1:] != chromoB[nB+1:])
  
  
def remove_homologous_pairs(contact_dict):

  from collections import defaultdict
  new_contacts = defaultdict(dict)
  
  for (chromoA, chromoB), contacts in flatten_dict(contact_dict).items():
    if not homologous_pair(chromoA, chromoB):
      new_contacts[chromoA][chromoB] = contacts
  
  return dict(new_contacts)


DEFAULT_DISAMBIG_MIN = 3
DEFAULT_DISAMBIG_FRAC = 0.02

def resolve_homolog_ambiguous(contact_dict, min_disambig=DEFAULT_DISAMBIG_MIN, frac_disambig=DEFAULT_DISAMBIG_FRAC):

  from collections import defaultdict

  group_sizes = defaultdict(int)
  chromo_pair_groups = defaultdict(set)

  # Consider all trans chromosome pairs
  # If the pair has fewer than min_disambig unambigous contacts it can have no homologous chromosome ambigous contacts
  # This filtering is only for homologous chromosome ambiguity, not positional ambiguity

  # Fetch the sizes of all ambiguity groups and which group goes with which chromo pair
  for chr_a in contact_dict:  
    for chr_b in contact_dict[chr_a]:    
      contacts = contact_dict[chr_a][chr_b].T

      for contact in contacts:
        ambig_group = contact['ambiguity']
        group_sizes[ambig_group] += 1

        if chr_a > chr_b:
          chr_a, chr_b = chr_b, chr_a

        chromo_key = (chr_a, chr_b)
        chromo_pair_groups[chromo_key].add(ambig_group)

  n_groups = len(group_sizes)

  # Count total unambiguous groups for each chromo pair
  pair_stats = {}
  unambig_fracs = []
  unambig_counts = []
  for chromo_key in chromo_pair_groups:
    pair_groups = chromo_pair_groups[chromo_key]
    n_unambig = 0

    for ambig_group in pair_groups:
      count = group_sizes[ambig_group]

      if count == 1:
        n_unambig += 1

    frac_unambig = n_unambig/float(len(pair_groups))

    if 0.0 < frac_unambig < 1.0:
      unambig_counts.append(n_unambig)
      unambig_fracs.append(frac_unambig)

    pair_stats[chromo_key] = (n_unambig, frac_unambig)

  # Calculate threshold statistic and excluded pairs

  p90 = np.percentile(unambig_counts, [90.0])[0] # Threshold to seek only chromo pairs where counts are high
  median_frac = np.median([unambig_fracs[i] for i, v in enumerate(unambig_counts) if v >= p90]) # An estimate of the average unambiguous proportion
  non_contact_chromos = set()

  for chromo_key in pair_stats:
    n_unambig, frac_unambig = pair_stats[chromo_key]
    chr_a, chr_b = chromo_key

    if (n_unambig < min_disambig) or (frac_unambig < (frac_disambig * median_frac)):
      chr_a, chr_b = chromo_key

      if chr_a != chr_b:
        non_contact_chromos.add(chromo_key)

  new_contact_dict = defaultdict(dict)

  for chr_a in contact_dict:  
    for chr_b in contact_dict[chr_a]:    
      contacts = contact_dict[chr_a][chr_b].T

      contacts_new = []
      for contact in contacts:
        ambig_group = contact['ambiguity']
        group_size = group_sizes[ambig_group]

        if chr_a > chr_b:
          chr_a, chr_b = chr_b, chr_a

        chromo_key = (chr_a, chr_b)
        if chromo_key in non_contact_chromos:
          if group_size == 1: # Keep unambiguous for excluded pairs unless homologs
            not_homolog = chr_a.split('.')[0] != chr_b.split('.')[0]

            if not_homolog:
              contacts_new.append(contact)
        else:
          contacts_new.append(contact)

      if contacts_new:
        contacts = np.array(contacts_new, dtype=Contact)
        new_contact_dict[chr_a][chr_b] = np.array(contacts, dtype= Contact)

  return new_contact_dict
  

def resolve_3d_ambiguous(contact_dict, seq_pos_dict, coords_dict, percentile_threshold=98.0, min_bead_sep=3):
  
  from collections import defaultdict
  
  ambig_groups = defaultdict(list)
  
  contact_pos = defaultdict(list)
  
  for chr_a in contact_dict:
    if chr_a not in seq_pos_dict:
      continue
  
    for chr_b in contact_dict[chr_a]:
      if chr_b not in seq_pos_dict:
        continue
    
      contacts = contact_dict[chr_a][chr_b].T

      for contact in contacts:
        pos_a, pos_b = contact['pos']
        ambig_group = contact['ambiguity']
        # None vals below will be filled with structure distances and bead separations
        ambig_groups[ambig_group].append([chr_a, pos_a, chr_b, pos_b, None, None])
        contact_pos[chr_a].append(pos_a)
        contact_pos[chr_b].append(pos_b)
        
  ambig_groups = {g:ambig_groups[g] for g in ambig_groups if len(ambig_groups[g]) > 1}

  info('Get contacts particle indices')
  
  pos_idx = {}
  
  for chromo in contact_pos:
    if chromo in coords_dict:
      pos_idx[chromo] = cps = {}
 
      seq_pos = seq_pos_dict[chromo]
      j_max = len(seq_pos)-1
      idx = np.searchsorted(seq_pos, contact_pos[chromo], side='right')
 
      for i, pos in enumerate(contact_pos[chromo]):
        j = min(j_max, idx[i])
        cps[pos] = j
  
  info('Get distances')
  
  closest_nz = []
  min_dists = {}
  min_seps = {}
  
  for ambig_group in ambig_groups:
    dists = []
    bead_seps = []
    
    for vals in ambig_groups[ambig_group]:
      chr_a, pos_a, chr_b, pos_b, null, null2 = vals  
      
      coords_a = coords_dict[chr_a]
      coords_b = coords_dict[chr_b]
      
      i = pos_idx[chr_a][pos_a]
      j = pos_idx[chr_b][pos_b]
      
      deltas = coords_a[:,i] - coords_b[:,j]
      dists2 = (deltas * deltas).sum(axis=1)
      
      dist = np.sqrt(dists2.mean())
 
      dists.append(dist)
      
      if chr_a == chr_b:
        bead_sep = abs(j-i)
      else:
        bead_sep = 99999 # Just something huge for trans
      
      bead_seps.append(bead_sep)
      
      vals[4] = dist
      vals[5] = bead_sep
    
    min_sep = min(bead_seps)
    min_seps[ambig_group] = min_sep
      
    min_dist = min(dists)
    min_dists[ambig_group] = min_dist
    if min_dist > 0.0:
      closest_nz.append(min_dist)
  
  remove = set()
  
  if closest_nz:
    dist_thresh = np.percentile(closest_nz, [percentile_threshold])[0]
  
    info('Filter ambiguity groups. Threshold = %.2f' % dist_thresh)
  
    for ambig_group in ambig_groups:
      min_dist = min_dists[ambig_group]
      min_sep = min_seps[ambig_group]

      for chr_a, pos_a, chr_b, pos_b, dist, bead_sep in ambig_groups[ambig_group]:
        if (min_dist == 0) and (chr_a != chr_b):
          remove.add((ambig_group, chr_a, pos_a, chr_b, pos_b))
          continue
        
        if (min_sep < min_bead_sep) and (chr_a != chr_b):
          remove.add((ambig_group, chr_a, pos_a, chr_b, pos_b))
          continue
        
        if dist != min_dist:
          if dist > min_dist + dist_thresh/2:
            remove.add((ambig_group, chr_a, pos_a, chr_b, pos_b))
            continue
          
        if dist > dist_thresh:
          remove.add((ambig_group, chr_a, pos_a, chr_b, pos_b))
          continue

  new_contact_dict = defaultdict(dict)

  for (chromoA, chromoB), contacts in flatten_dict(contact_dict).items():
    contacts = [c for c in contacts if (c['ambiguity'], chromoA, c['pos'][0], chromoB, c['pos'][1]) not in remove]
    if contacts:
      new_contact_dict[chromoA][chromoB] = np.array(contacts, dtype=Contact)
  
  return dict(new_contact_dict)


def between(x, a, b):
  return (a < x) & (x < b)


def initial_clean_contacts(contact_dict, threshold_cis=int(2e6), threshold_trans=int(10e6), pos_error=100, ignore=()):
  """
  Select only unambigous contacts that are within a given sequence separation of another
  contact, for the same chromosome pair.
  """
  
  info('Cleaning initial contact list')
  
  from numpy import array, zeros, eye
  from collections import defaultdict
  new_contacts = defaultdict(dict)
  ag_size = defaultdict(int)
  
  contact_pos = defaultdict(list)
  
  for chr_pair, contacts in flatten_dict(contact_dict).items():
    for ag in contacts['ambiguity']:
      ag_size[ag] += 1
        
  for (chromoA, chromoB), contacts in flatten_dict(contact_dict).items():
    # Select only unambiguous
    unambig = np.array([ag_size[ag] == 1 for ag in contacts['ambiguity']])
    contacts = contacts[unambig]
    
    if chromoA == chromoB:
      threshold = threshold_cis
    else:
      threshold = threshold_trans
      
    
    # positions is N x 2 matrix, where there are N contacts (and the 2 is because of pos_a, pos_b)
    positions = contacts['pos'].astype('int32')
    if chromoA in ignore or chromoB in ignore:
      new_contacts[chromoA][chromoB] = contacts
      continue
      
    if len(positions) == 0: # Sometimes empty e.g. for MT, Y chromos
      continue
    # positions[:, 0] and positions[:, 1] have shape N
    # the [None, ...] converts that into shape 1 x N (but it looks like you don't need the ...)
    pos_a = positions[:, 0][None, ...]
    pos_b = positions[:, 1][None, ...]
    nondiagonal = ~eye(len(positions), dtype='bool') # False on diagonal, True off diagonal

    supported = zeros(len(positions), dtype='bool')
    # do not need nondiagonal in first expression below as long as pos_error > 0
    # the diagonal in the second expression so as not to include i if a_i happens to be close to b_i
    # abs(pos_a - pos_b.T) gives N x N matrix with difference in positions between a_i and b_j
    # we want both ends in a contact to be close to the ends in another contact
    supported |= (between(abs(pos_a - pos_a.T), pos_error, threshold) &
                  between(abs(pos_b - pos_b.T), pos_error, threshold)).any(axis=0)
    supported |= (between(abs(pos_a - pos_b.T), pos_error, threshold) &
                  between(abs(pos_b - pos_a.T), pos_error, threshold) &
                  nondiagonal).any(axis=0)
    
    if not supported.any():
      continue
    
    new_contacts[chromoA][chromoB] = contacts[supported]
    
  return dict(new_contacts)


def remove_violated_contacts(contact_dict, coords_dict, particle_seq_pos, threshold=5.0):
  """
  Remove contacts whith structure distances that exceed a given threshold
  """
  from numpy import int32, sqrt, array
  from collections import defaultdict
  new_contacts = defaultdict(dict)

  for chr_a in contact_dict:
    if chr_a not in coords_dict:
      continue
    for chr_b, contacts in contact_dict[chr_a].items():
      if chr_b not in coords_dict:
        continue

      contact_pos_a = contacts['pos'][:, 0].astype(int32)
      contact_pos_b = contacts['pos'][:, 1].astype(int32)

      coords_a = coords_dict[chr_a]
      coords_b = coords_dict[chr_b]
      
      struc_dists = []

      for m in range(len(coords_a)):
        coord_data_a = get_interpolated_coords(coords_a[m], contact_pos_a, particle_seq_pos[chr_a])
        coord_data_b = get_interpolated_coords(coords_b[m], contact_pos_b, particle_seq_pos[chr_b])

        deltas = coord_data_a - coord_data_b
        dists = sqrt((deltas*deltas).sum(axis=1))
        struc_dists.append(dists)

      # Average over all conformational models
      struc_dists = array(struc_dists).T.mean(axis=1)

      # Select contacts with distances below distance threshold
      indices = (struc_dists < threshold).nonzero()[0]
      new_contacts[chr_a][chr_b] = contacts[indices]

  return dict(new_contacts)
  
  
def get_random_coords(shape, radius=10.0):
  """
  Get random, uniformly sampled coorinate positions, restricted to
  a nD-sphere of given radius
  """

  from numpy import random
  from numpy.linalg import norm

  u = random.uniform(size=shape[:-1])
  x = random.normal(size=shape)
  scaling = (radius * u ** (1/shape[-1])) / norm(x, axis=-1)
  return scaling[..., None] * x


def get_interpolated_coords(coords, pos, prev_pos):
  from numpy import interp, apply_along_axis
  from functools import partial
  
  coords = apply_along_axis(partial(interp, pos, prev_pos), -2, coords)
   
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


def anneal_model(model_data, anneal_schedule, masses, radii, restraints, rep_dists,
                 ambiguity, temp, time_step, dyn_steps, repulse, dist, bead_size):
                             
  import gc
  
  m, model_coords = model_data
  
  # Anneal one model in parallel
  
  time_taken = 0.0
  
  if m == 0:
    print_interval = max(1, dyn_steps/2)
  
  else:
    print_interval = 0
  
  info('  starting model %d' % m)
  
  nrep_max = np.int32(0)
  
  for temp, repulse in anneal_schedule:
    gc.collect() # Try to free some memory
    
    # Update coordinates for this temp
    
    try:
      dt, nrep_max = dyn_util.run_dynamics(model_coords, masses, radii, rep_dists,
                                         restraints['indices'], restraints['dists'],
                                         restraints['weight'], ambiguity,
                                         temp, time_step, dyn_steps, repulse, dist,
                                         bead_size, nrep_max,
                                         print_interval=print_interval)
    
    except Exception as err:
      print(f"Error while running dynamics: {err}")
      break 
    #  return err
    
    nrep_max = np.int32(nrep_max * 1.2)
    time_taken += dt
  
  # Center
  model_coords -= model_coords.mean(axis=0)

  info('  done model %d' % m)
  
  return model_coords
 

def calc_bins(chromo_limits, particle_size):
  from numpy import arange
  from math import ceil

  bins = {}
  for chromo, (start, end) in chromo_limits.items():
    first_bin = start // particle_size
    last_bin = 1 + (end-1) // particle_size
    start = particle_size * first_bin
    end = particle_size * last_bin
    bins[chromo] = arange(start, end, particle_size, dtype='int32')
    
  return bins


def calc_ambiguity_offsets(groups):
  """
  Convert (sorted) ambiguity groups to group-offsets for
  annealing calculations.
  """
  from numpy import flatnonzero, zeros

  group_starts = zeros(len(groups) + 1, dtype='bool')
  group_starts[-1] = group_starts[0] = 1
  group_starts[:-1] |= groups == 0
  group_starts[1:-1] |= groups[1:] != groups[:-1]
  
  return flatnonzero(group_starts).astype('int32')


def backbone_restraints(seq_pos, particle_size, scale=1.0, lower=0.1, upper=1.1, weight=1.0):
  from numpy import empty, arange, array

  restraints = empty(len(seq_pos) - 1, dtype=Restraint)
  offsets = array([0, 1], dtype='int')
  restraints['indices'] = arange(len(restraints))[:, None] + offsets

  # Normally 1.0 for regular sized particles
  bounds = array([lower, upper], dtype='float') * scale
  restraints['dists'] = ((seq_pos[1:] - seq_pos[:-1]) / particle_size)[:, None] * bounds
  restraints['ambiguity'] = 0 # Use '0' to represent no ambiguity
  restraints['weight'] = weight

  return restraints


def flatten_dict(d):

  r = {}
  for key, value in d.items():
    if isinstance(value, dict):
      r.update({(key,) + k: v for k, v in flatten_dict(value).items()})
    else:
      r[(key,)] = value
  return r


def tree():
  from collections import defaultdict
  
  def tree_():
    return defaultdict(tree_)
    
  return tree_()


def unflatten_dict(d):

  r = tree()

  for ks, v in d.items():
    d = r
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = v
    
  return r


def concatenate_restraints(restraint_dict, pos_dict):
  from itertools import accumulate, chain
  from functools import partial
  from numpy import concatenate, array, empty
  import operator as op

  chromosomes = sorted(pos_dict)
  chromosome_offsets = dict(zip(chromosomes, accumulate(chain(
    [0], map(len, map(partial(op.getitem, pos_dict), chromosomes)))
  )))
  
  flat_restraints = sorted(flatten_dict(restraint_dict).items())

  r = concatenate(list(map(op.itemgetter(1), flat_restraints)))
  r['indices'] = concatenate([
    restraints['indices'] + array([[chromosome_offsets[c] for c in chrs]])
    for chrs, restraints in flat_restraints
  ])

  return r


def calc_restraints(contact_dict, pos_dict, particle_size,
                    scale=1.0, lower=0.8, upper=1.2, weight=1.0):
  from numpy import empty, array, searchsorted
  from collections import defaultdict, Counter

  dists = array([[lower, upper]]) * scale

  r = defaultdict(dict)
  for (chr_a, chr_b), contacts in flatten_dict(contact_dict).items():
    if len(contacts) < 1:
      continue
    r[chr_a][chr_b] = empty(len(contacts), dtype=Restraint)

    # below is not quite correct if pos_dict[chr_a] value is exactly the same as contacts['pos'][:, 0]
    # (and similarly for chr_b) because then you don't want to subtract 1
    idxs_a = searchsorted(pos_dict[chr_a], contacts['pos'][:, 0], side='right') - 1 # -1 because want bin that is floor
    idxs_b = searchsorted(pos_dict[chr_b], contacts['pos'][:, 1], side='right') - 1
    # pos_dict[chromo] gives the bin positions so pos_dict[chromo][0] gives the first bin position
    #idxs_a = (contacts['pos'][:, 0] - pos_dict[chr_a][0]) // particle_size  # bin number
    #idxs_b = (contacts['pos'][:, 1] - pos_dict[chr_b][0]) // particle_size
    r[chr_a][chr_b]['indices'] = array([idxs_a, idxs_b]).T
    r[chr_a][chr_b]['ambiguity'] = contacts['ambiguity']
    r[chr_a][chr_b]['dists'] = dists
    r[chr_a][chr_b]['weight'] = weight
    
  return dict(r)


def bin_restraints(restraints):
  from numpy import unique, bincount, empty, concatenate, sort, copy
  restraints = copy(restraints)
  restraints['indices'] = sort(restraints['indices'], axis=1)

  # Ambiguity group 0 is unique restraints, so they can all be binned
  _, inverse, counts = unique(restraints['ambiguity'],
                              return_inverse=True,
                              return_counts=True)
                              
  restraints['ambiguity'][counts[inverse] == 1] = 0

  # Bin within ambiguity groups
  uniques, idxs = unique(restraints[['ambiguity', 'indices', 'dists']],
                         return_inverse=True)

  binned = empty(len(uniques), dtype=Restraint)
  binned['ambiguity'] = uniques['ambiguity']
  binned['indices'] = uniques['indices']
  binned['dists'] = uniques['dists']
  binned['weight'] = bincount(idxs, weights=restraints['weight'])
  
  return binned


def merge_dicts(*dicts):
  from collections import defaultdict
  from numpy import empty, concatenate

  new = defaultdict(list)
  for d in dicts:
    for k, v in flatten_dict(d).items():
      new[k].append(v)
      
  new = {k: concatenate(v) for k, v in new.items()}
  
  return unflatten_dict(new)


def determine_bead_size(particle_size):
  
  return particle_size ** (1/3)


def anneal_genome(contact_dict, num_models, particle_size,
                  general_calc_params, anneal_params,
                  prev_seq_pos_dict=None, start_coords=None, num_cpu=MAX_CORES):
  """
  Use chromosome contact data to generate distance restraints and then
  apply a simulated annealing protocul to generate/refine coordinates.
  Starting coordinates may be random of from a previous (e.g. lower
  resolution) stage.
  """
  
  from numpy import (int32, ones, empty, random, concatenate, stack, argsort, arange,
                     arctan, full, zeros)
  from math import log, exp, atan, pi
  from functools import partial
  import gc

  bead_size = determine_bead_size(particle_size)

  random.seed(general_calc_params['random_seed'])
  
  particle_size = int32(particle_size)
  seq_pos_dict = calc_bins(calc_limits(contact_dict), particle_size)
  
  chromosomes = sorted(set.union(set(contact_dict), *map(set, contact_dict.values())))

  chrs = list(chromosomes)
  for chromo in chrs:
    if chromo not in seq_pos_dict:
      chromosomes.remove(chromo)
      
  points = chromosomes[:]

  restraint_dict = calc_restraints(contact_dict, seq_pos_dict, particle_size,
                                   scale=bead_size,
                                   lower=general_calc_params['contact_dist_lower'],
                                   upper=general_calc_params['contact_dist_upper'],
                                   weight=1.0)

  # Adjust to keep force/particle approximately constant
  # why 215 (in opencl version this is 430 instead)
  dist = 215.0 * (sum(map(len, seq_pos_dict.values())) /
                  sum(map(lambda v: v['weight'].sum(),
                          flatten_dict(restraint_dict).values())))

  restraint_dict = merge_dicts(
    restraint_dict,
    # Backbone restraints
    {chromo: {chromo: backbone_restraints(
      seq_pos_dict[chromo], particle_size, bead_size,
      lower=general_calc_params['backbone_dist_lower'],
      upper=general_calc_params['backbone_dist_upper'], weight=1.0
    )} for chromo in chromosomes}
  )

  coords = start_coords or {}
  for chromo in chromosomes:
    if chromo not in coords:
      coords[chromo] = get_random_coords((num_models, len(seq_pos_dict[chromo]), 3),
                                         general_calc_params['random_radius'] * bead_size)
                                         
    elif coords[chromo].shape[1] != len(seq_pos_dict[chromo]):
      coords[chromo] = get_interpolated_coords(coords[chromo], seq_pos_dict[chromo],
                                               prev_seq_pos_dict[chromo])

  # Equal unit masses and radii for all particles
  masses = {chromo:ones(len(pos), float) for chromo, pos in seq_pos_dict.items()}
  radii = {chromo:full(len(pos), bead_size, float) for chromo, pos in seq_pos_dict.items()}
  rep_dists = {chromo: r * 1.0 for chromo, r in radii.items()}

  # Concatenate chromosomal data into a single array of particle restraints
  # for structure calculation.
  restraints = bin_restraints(concatenate_restraints(restraint_dict, seq_pos_dict))
  coords = concatenate([coords[chromo] for chromo in points], axis=1)
  masses = concatenate([masses[chromo] for chromo in points])
  radii = concatenate([radii[chromo] for chromo in points])
  rep_dists = concatenate([rep_dists[chromo] for chromo in points])

  restraint_order = argsort(restraints['ambiguity'])
  restraints = restraints[restraint_order]
  ambiguity = calc_ambiguity_offsets(restraints['ambiguity'])
  
  # Annealing parameters
  temp_start = anneal_params['temp_start']
  temp_end = anneal_params['temp_end']
  temp_steps = anneal_params['temp_steps']
  
  # Setup annealing schedule: setup temps and repulsive terms
  adj = 1.0 / atan(10.0)
  decay = log(temp_start/temp_end)    
  anneal_schedule = []
  
  for step in range(temp_steps):
    gc.collect() # Try to free some memory
    
    frac = step/float(temp_steps)
  
    # exponential temp decay
    temp = temp_start * exp(-decay*frac)
    
    # sigmoidal repusion scheme
    repulse = 0.5 + adj * atan(frac*20.0-10) / pi 
    
    temp *= bead_size ** 2
    repulse /= bead_size ** 2
    
    anneal_schedule.append((temp, repulse))  
      
  # Paricle dynamics parameters
  # (these need not be fixed for all stages, but are for simplicity)    
  dyn_steps = anneal_params['dynamics_steps']
  time_step = anneal_params['time_step']
     
  # Update coordinates in the annealing schedule which is applied to each model in parallel
  common_args = [anneal_schedule, masses, radii, restraints, rep_dists,
                 ambiguity, temp, time_step, dyn_steps, repulse, dist, bead_size]
  
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
    file_obj = open(file_path, mode or 'r', IO_BUFFER)
 
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


def svd_rotate(coords_a, coords_b, weights=None):
  """
  Aligns coords_b to coords_a by rotation, returning transformed coords
  """
  
  coords_bt = coords_b.transpose()
  
  if weights is None:
    coords_bt = coords_b.transpose()
  else:
    coords_bt = coords_b.transpose() * weights

  mat = np.dot(coords_bt, coords_a)
  
  try:
    rot_mat1, _scales, rot_mat2 = np.linalg.svd(mat)
  
  except np.linalg.LinAlgError as err:
    return coords_b
    
  sign = np.linalg.det(rot_mat1) * np.linalg.det(rot_mat2)

  if sign < 0:
    rot_mat1[:,2] *= -1
  
  rotation = np.dot(rot_mat1, rot_mat2)
  
  return np.dot(coords_b, rotation)


def calc_rmsds(ref_coords, coord_models, weights=None):
  """
  Calculates per model and per particle RMSDs compared to reference coords
  """
  
  n_coords = len(ref_coords)
  n_models = len(coord_models)
  
  if weights is None:
    weights = np.ones(n_coords)
  
  model_rmsds = []
  sum_weights = sum(weights)
  sum_deltas2 = np.zeros((n_coords, 3))
  
  for coords in coord_models:
    deltas2 = (coords-ref_coords)**2
    sum_deltas2 += deltas2
    dists2 = weights*deltas2.sum(axis=1)
    model_rmsds.append(np.sqrt(sum(dists2))/sum_weights)
  
  particle_rmsds = np.sqrt(sum_deltas2.sum(axis=1)/n_models)

  return model_rmsds, particle_rmsds
  
  
def center_coords(coords, weights):
  """
  Transpose coords to zero at centroid
  """
  
  wt_coords = coords.transpose() * weights
  xyz_totals = wt_coords.sum(axis=1)
  center = xyz_totals/sum(weights)
  cen_coords = coords - center
  
  return cen_coords


def align_coord_pair(coords_a, coords_b, dist_scale=1.0):
  """
  Align two coord arrays.
  Returns the transformed version of coords_a and coords_b.
  Returns the model and particle RMSD.
  """
  
  n = len(coords_a)
  weights = np.ones(n, float) # All particles intially weighted equally
  
  # Move both coords arrays to origin, original inputs are preserved
  coords_a = center_coords(coords_a, weights)
  coords_b = center_coords(coords_b, weights)
  
  # Align B to A and get RMDs of transformed coords
  coords_b1 = svd_rotate(coords_a, coords_b, weights)
  rmsd_1, particle_rmsds_1 = calc_rmsds(coords_a, [coords_b1], weights)

  # Align mirror B to A and get RMDs of transformed coords
  coords_b2 = svd_rotate(coords_a, -coords_b, weights)
  rmsd_2, particle_rmsds_2 = calc_rmsds(coords_a, [coords_b2], weights)
  
  if rmsd_1[0] < rmsd_2[0]:
    coords_b = coords_b1
    particle_rmsds = particle_rmsds_1
    
  else: # Mirror is best
    coords_b = coords_b2
    particle_rmsds = particle_rmsds_2
        
  # Refine alignment with exponential weights that deminish as RMSD increases
  if dist_scale:
    med_rmsd = np.median(particle_rmsds) # Gives a degree of scale invariance
    weight_scale = particle_rmsds / dist_scale
    weights_exp = np.exp(-weight_scale*weight_scale*med_rmsd)

    coords_a = center_coords(coords_a, weights_exp)
    coords_b = center_coords(coords_b, weights_exp)
    coords_b = svd_rotate(coords_a, coords_b, weights_exp)
    
  return coords_a, coords_b
  
  
def align_coord_models(coord_models, n_iter=1, dist_scale=True):
  """
  Aligns multiple coords arrays.
  Convergence is normally good with n_iter=1, but larger values may be needed in troublesome cases.
  Set the dist_scale to down-weight RMSD values that approach/exceed this size.
  The default dist_scale(=True) automatically sets a value to only ignore the worst outliers.
  Setting dist_scale to None or any false value disables all RMSD based weighting; this can give a better overall RMSD,
  at it tries to fit outliers, but worse alignment for the most invariant particles.
  Returns aligned coords models, RMSD of each model and RMSD of each particle relative to mean.
  """

  coord_models = np.array(coord_models)
  n_models, n_coords = coord_models.shape[:2]
      
  if dist_scale is True:
    init_dist_scale = 0.0
  else:
    init_dist_scale = dist_scale
  
  # Align to first model arbitrarily
  ref_coords = coord_models[0]
  for i, coords in enumerate(coord_models[1:], 1):
    coords_a, coords_b = align_coord_pair(ref_coords, coords, init_dist_scale)
    coord_models[i] = coords_b
    
  coord_models[0] = coords_a # First model has been centred

  # Align all coord models to closest to mean (i.e. a real model)
  # given initial mean could be wonky if first model was poor
  model_rmsds, particle_rmsds = calc_rmsds(coord_models.mean(axis=0), coord_models)

  # When automated the distance scale is set according to a large RMSD vale
  if dist_scale is True:
    dist_scale = np.percentile(particle_rmsds, [99.0])[0]
    
  j = np.array(model_rmsds).argmin()
  ref_coords = coord_models[j]
  
  for i, coords in enumerate(coord_models):
    if i != j:
      coords_a, coords_b = align_coord_pair(ref_coords, coords, dist_scale)
      coord_models[i] = coords_b
  
  # Align all coord models to mean and converge iteratively
  for j in range(n_iter):
    ref_coords = coord_models.mean(axis=0)
    for i, coords in enumerate(coord_models):
      coords_a, coords_b = align_coord_pair(ref_coords, coords, dist_scale)
      coord_models[i] = coords_b
      
  # Final mean for final RMSDs
  model_rmsds, particle_rmsds = calc_rmsds(coord_models.mean(axis=0), coord_models)
  
  return coord_models, model_rmsds, particle_rmsds
  
  
def align_chromo_models(coords_dict, seq_pos_dict):
  
  coord_models = None
  for chromo in seq_pos_dict:
    chromo_coords = coords_dict[chromo]
    # chromo_coords is of shape ()#models, #beads, #dim=3)
    if coord_models is None:
      coord_models = coords_dict[chromo]
    else:
      coord_models= np.concatenate((coord_models, coords_dict[chromo]), axis=1)
  
  coord_models, model_rmsds, particle_rmsds = align_coord_models(coord_models)
  
  return coord_models, model_rmsds, particle_rmsds
  

def convert_models_to_dict(coord_models, coords_dict, seq_pos_dict):
  
  n = 0
  for chromo in seq_pos_dict:
    m = coords_dict[chromo].shape[1]  # the number of beads in this chromo
    coords_dict[chromo] = coord_models[:, n:n+m, :]
    n += m

  return coords_dict
  
def align_chromo_coords(coords_dict, seq_pos_dict):
      
  coord_models, model_rmsds, particle_rmsds = align_chromo_models(coords_dict, seq_pos_dict)
    
  coords_dict = convert_models_to_dict(coord_models, coords_dict, seq_pos_dict)
  
  return coords_dict
  
def remove_models(coords_dict, seq_pos_dict, req_num_models):
      
  from numpy import array

  coord_models, model_rmsds, particle_rmsds = align_chromo_models(coords_dict, seq_pos_dict)
  
  model_rmsds_indices = [(model_rmsd, n) for (n, model_rmsd) in enumerate(model_rmsds)]
  model_rmsds_indices.sort()
  indices = array([ind for (rmsd, ind) in model_rmsds_indices])
  indices = indices[:req_num_models]
  coord_models = coord_models[indices]
  
  coords_dict = convert_models_to_dict(coord_models, coords_dict, seq_pos_dict)
  
  return coords_dict

def export_coords(out_format, out_file_path, coords_dict, particle_seq_pos, particle_size, num_models):
  
  # Save final coords as N3D or PDB format file
  
  bead_size = determine_bead_size(particle_size)
  coords_dict_scaled = {}
  for chromo in coords_dict:
    coords_dict_scaled[chromo] = coords_dict[chromo] / bead_size
  
  if num_models > 1:
    coords_dict_scaled = align_chromo_coords(coords_dict_scaled, particle_seq_pos)
    
  if out_format == PDB:
    if not out_file_path.endswith(PDB):
      out_file_path = '%s.%s' % (out_file_path, PDB)
  
    export_pdb_coords(out_file_path, coords_dict_scaled, particle_seq_pos, particle_size)
 
  else:
    if not out_file_path.endswith(N3D):
      out_file_path = '%s.%s' % (out_file_path, N3D)
      
    export_n3d_coords(out_file_path, coords_dict_scaled, particle_seq_pos)
    
  info('Saved structure file to: %s' % out_file_path)


def contact_count(contact_dict):
  
  count = 0
  for contacts in flatten_dict(contact_dict).values():
    count += len(contacts)
    
  return count
  
  
def particle_size_file_path(file_path, particle_size):
  
  file_root, file_ext = os.path.splitext(file_path)  
  size = int(particle_size/1000)
  file_path = '%s_%d%s' % (file_root, size, file_ext)
   
  return file_path


def remove_violated_threshold(particle_size, bead_size):
  
  threshold = None
  if particle_size < 0.25e6:
    threshold = 5.0 * bead_size
  elif particle_size < 0.5e6:
    threshold = 6.0 * bead_size
  
  return threshold
  
def calc_genome_structure(ncc_file_path, out_file_path, general_calc_params, anneal_params,
                          particle_sizes, num_models=5, isolation_threshold_cis=int(2e6),
                          isolation_threshold_trans=int(10e6), out_format=N3D, num_cpu=MAX_CORES,
                          start_coords_path=None, save_intermediate=False, save_intermediate_ncc=False,
                          have_diploid=False, struct_ambig_size=DEFAULT_STRUCT_AMBIG_SIZE,
                          remove_models_size=DEFAULT_REMOVE_MODELS_SIZE):

  from time import time
  
  req_num_models = num_models
  
  # Load single-cell Hi-C data from NCC contact file, as output from NucProcess
  using_inactive = False
  contact_dict = load_ncc_file(ncc_file_path)
  info('Total number of contacts = %d' % contact_count(contact_dict))

  #if have_diploid:
  #  contact_dict = resolve_homolog_ambiguous(contact_dict)
  #  info('Total number of contacts after resolving homologous ambiguity = %d' % contact_count(contact_dict))
  #  if save_intermediate_ncc:
  #    save_ncc_file(ncc_file_path, 'resolve_homo', contact_dict, particle_sizes[0])

  if start_coords_path:
    prev_seq_pos, start_coords = load_n3d_coords(start_coords_path)
    if start_coords:
      chromo = next(iter(start_coords)) # picks out arbitrary chromosome
      num_models = len(start_coords[chromo])
  else:
    # Record particle positions from previous stages
    # so that coordinates can be interpolated to higher resolution
    prev_seq_pos = None
    start_coords = None  # Initial coords will be random
    
  for stage, particle_size in enumerate(particle_sizes):
    info("Running structure calculation stage %d (%d kb)" % (stage+1, particle_size//1000))
    
    if particle_size <= struct_ambig_size and not using_inactive:
      contact_dict = load_ncc_file(ncc_file_path, active_only=False)
      using_inactive = True
      
    stage_contact_dict = dict(contact_dict) # Each stage reconsiders all contacts

    anneal_params['temp_steps'] = anneal_params['temp_steps_by_stage'][stage]
    anneal_params['dynamics_steps'] = anneal_params['dynamics_steps_by_stage'][stage]

    # Can remove large violations (noise contacts inconsistent with structure)
    # once we have a reasonable resolution structure
    bead_size = determine_bead_size(particle_size)
    
    if start_coords is None:
      # Only use unambiguous contacts which are supported by others nearby in sequence, in the initial instance
      stage_contact_dict = initial_clean_contacts(stage_contact_dict, threshold_cis=isolation_threshold_cis,
                                                    threshold_trans=isolation_threshold_trans)
      
      if save_intermediate_ncc:
        save_ncc_file(ncc_file_path, 'remove_isol', stage_contact_dict, particle_size)
      
      if particle_size > remove_models_size:
        num_models = 2 * req_num_models
      else:
        num_models = req_num_models
      
    else:
      if particle_size <= remove_models_size and num_models > req_num_models:
        start_coords = remove_models(start_coords, prev_seq_pos, req_num_models)
        chromo = next(iter(start_coords)) # picks out arbitrary chromosome
        num_models = len(start_coords[chromo])
        info('Removed %d models. Calculating %d' % (num_models-req_num_models, req_num_models))
        
      if particle_size <= struct_ambig_size:
        stage_contact_dict = resolve_3d_ambiguous(stage_contact_dict, prev_seq_pos, start_coords)
        info('Total number of contacts after resolving ambiguity with structure = %d' % contact_count(stage_contact_dict))
        
        if save_intermediate_ncc:
          save_ncc_file(ncc_file_path, 'resolve_3d', stage_contact_dict, particle_size)
          
      else:
        # Only use supported, unambiguous contacts
        stage_contact_dict = initial_clean_contacts(stage_contact_dict, threshold_cis=isolation_threshold_cis,
                                                    threshold_trans=isolation_threshold_trans)
        if save_intermediate_ncc:
           save_ncc_file(ncc_file_path, 'remove_isol', stage_contact_dict, particle_size)
            
    threshold = remove_violated_threshold(particle_size, bead_size)
    
    if threshold and start_coords:
      stage_contact_dict = remove_violated_contacts(stage_contact_dict, start_coords, prev_seq_pos,
                                                    threshold=threshold)
      info('Total number of contacts after removing violated contacts = %d' % contact_count(stage_contact_dict))
      if save_intermediate_ncc:
        save_ncc_file(ncc_file_path, 'remove_viol', stage_contact_dict, particle_size)

    coords_dict, particle_seq_pos = anneal_genome(stage_contact_dict, num_models, particle_size,
                                                  general_calc_params, anneal_params,
                                                  prev_seq_pos, start_coords, num_cpu)

    if save_intermediate and stage < len(particle_sizes)-1:
      file_path = particle_size_file_path(out_file_path, particle_size)
      export_coords(out_format, file_path, coords_dict, particle_seq_pos, particle_size, num_models)
      
    # Next stage based on previous stage's 3D coords
    # and their respective seq. positions
    start_coords = coords_dict
    prev_seq_pos = particle_seq_pos

  # Save final coords
  file_path = particle_size_file_path(out_file_path, particle_size)
  export_coords(out_format, file_path, coords_dict, particle_seq_pos, particle_size, num_models)

  if particle_size <= struct_ambig_size:
    contact_dict = resolve_3d_ambiguous(contact_dict, particle_seq_pos, coords_dict)
    info('Final total number of contacts after removing structural ambiguity = %d' % contact_count(contact_dict))
    
    if save_intermediate_ncc:
      save_ncc_file(ncc_file_path, 'final', contact_dict, particle_size)


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
  isolation_threshold_cis = int(2e6)
  isolation_threshold_trans = int(10e6)
  
  calc_genome_structure(ncc_file_path, save_path, general_calc_params, anneal_params,
                        particle_sizes, num_models, isolation_threshold_cis,
                        isolation_threshold_trans, out_format='pdb')

test_imports()

import dyn_util

if __name__ == '__main__':
  
  import os, sys
  from time import time
  from argparse import ArgumentParser
  
  epilog = 'For further help on running this program please email tjs23@cam.ac.uk'
  
  arg_parse = ArgumentParser(prog=PROG_NAME, description=DESCRIPTION,
                             epilog=epilog, prefix_chars='-', add_help=True)

  arg_parse.add_argument('ncc_path', nargs=1, metavar='NCC_FILE',
                         help='Input NCC format file containing single-cell Hi-C contact data, e.g. use the demo data at example_chromo_data/Cell_1_contacts.ncc')

  arg_parse.add_argument('-o', metavar='OUT_FILE',
                         help='Optional name of output file for 3D coordinates in N3D or PDB format (see -f option). If not set this will be auto-generated from the input file name')

  arg_parse.add_argument('-logging', default=False, action='store_true',
                         help='Whether logging (to a file) is turned on')

  arg_parse.add_argument('-log_path', metavar='OUT_FILE',
                         help='Optional name of logging file. If not set this will be auto-generated from the input file name if logging is turned on')

  arg_parse.add_argument('-save_intermediate', default=False, action='store_true',
                         help='Write out intermediate coordinate files.')

  arg_parse.add_argument('-save_intermediate_ncc', default=False, action='store_true',
                         help='Write out intermediate NCC files.')

  arg_parse.add_argument('-start_coords_path', metavar='N3D_FILE',
                         help='Initial 3D coordinates in N3D format. If set this will override -m flag.')

  arg_parse.add_argument('-m', default=1, metavar='NUM_MODELS',
                         type=int, help='Number of alternative conformations to generate from repeat calculations with different random starting coordinates: Default: 1')

  arg_parse.add_argument('-f', metavar='OUT_FORMAT', default=N3D,
                         help='File format for output 3D coordinate file. Default: "%s". Also available: "%s"' % (N3D, PDB))

  arg_parse.add_argument('-s', nargs='+', default=[8.0,4.0,2.0,0.4,0.2,0.1], metavar='Mb_SIZE', type=float,
                         help='One or more sizes (Mb) for the hierarchical structure calculation protocol (will be used in descending order). Default: 8.0 4.0 2.0 0.4 0.2 0.1')

  arg_parse.add_argument('-diploid', default=False, action='store_true',
                         help='The data is (hybrid) diploid')

  arg_parse.add_argument('-struct_ambig_size', default=DEFAULT_STRUCT_AMBIG_SIZE_MB, metavar='Mb_SIZE', type=float,
                         help='The particle size (Mb) at or below which the ambiguous constraints (inc. for diploid data) are considered and filtered using the structure: Default %.1f' % DEFAULT_STRUCT_AMBIG_SIZE_MB)

  arg_parse.add_argument('-remove_models_size', default=DEFAULT_REMOVE_MODELS_SIZE_MB, metavar='Mb_SIZE', type=float,
                         help='The particle size (Mb) at or below which half the models are removed (based on RMSD): Default %.1f' % DEFAULT_REMOVE_MODELS_SIZE_MB)

  arg_parse.add_argument('-cpu', metavar='NUM_CPU', type=int, default=MAX_CORES,
                         help='Number of parallel CPU cores for calculating different coordinate models. Limited by the number of models (-m) but otherwise defaults to all available CPU cores (%d)' % MAX_CORES)

  arg_parse.add_argument('-iso_cis', default=2, metavar='Mb_SIZE', type=int,
                         help='Cis contacts must be near another, within this (Mb) separation threshold (at both ends) to be considered supported: Default 2.0')

  arg_parse.add_argument('-iso_trans', default=10, metavar='Mb_SIZE', type=int,
                         help='Trans contacts must be near another, within this (Mb) separation threshold (at both ends) to be considered supported: Default 10.0')

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

  arg_parse.add_argument('-temps_by_stage', default=[], metavar='NUM_STEPS', nargs='+',
                         type=int, help='Number of temperature steps (specified by hierarchical stage; see -s) in annealing protocol between start and end temperature. If this is specified then -temps argument ignored.')

  arg_parse.add_argument('-dyns', default=100, metavar='NUM_STEPS',
                         type=int, help='Number of particle dynamics steps to apply at each temperature in the annealing protocol. Default: 100')

  arg_parse.add_argument('-dyns_by_stage', default=[], metavar='NUM_STEPS', nargs='+',
                         type=int, help='Number of particle dynamics steps to apply (specified by hierarchical stage; see -s) at each temperature in the annealing protocol. If this is specified then -dyns argument ignored.')

  arg_parse.add_argument('-time_step', default=0.001, metavar='TIME_DELTA',
                         type=float, help='Simulation time step between re-calculation of particle velocities. Default: 0.001')
  
  args = vars(arg_parse.parse_args())
  
  ncc_file_path = args['ncc_path'][0]
  
  save_path = args['o']
  if save_path is None:
    save_path = os.path.splitext(ncc_file_path)[0]
     
  if args['logging']:
    log_path = args['log_path']
    if not log_path:
      log_path = '%s_log.txt' % save_path
    LOG_FILE_OBJ = open(log_path, 'w')
    
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
  temp_steps_by_stage = args['temps_by_stage']
  dynamics_steps = args['dyns']
  dynamics_steps_by_stage = args['dyns_by_stage']
  time_step = args['time_step']
  isolation_threshold_cis = args['iso_cis']
  isolation_threshold_trans = args['iso_trans']
  save_intermediate = args['save_intermediate']
  save_intermediate_ncc = args['save_intermediate_ncc']
  start_coords_path = args['start_coords_path']
  have_diploid = args['diploid']
  struct_ambig_size = args['struct_ambig_size']
  remove_models_size = args['remove_models_size']
  
  if out_format not in FORMATS:
    critical('Output file format must be one of: %s' % ', '.join(FORMATS))
  
  for val, name, sign in ((num_models,               'Number of conformational models', '+'),
                          (dist_power_law,           'Distance power law', '-0'),
                          (contact_dist_lower,       'Contact distance lower bound', '+'),
                          (contact_dist_upper,       'Contact distance upper bound', '+'),
                          (backbone_dist_lower,      'Backbone distance lower bound', '+'),
                          (backbone_dist_upper,      'Backbone distance upper bound', '+'),
                          (random_radius,            'Random-start radius', '+'),
                          (temp_start,               'Annealing start temperature', '+'),
                          (temp_end,                 'Annealing end temperature', '+0'),
                          (temp_steps,               'Number of annealing temperature steps', '+'),
                          (dynamics_steps,           'Number of particle dynamics steps', '+'),
                          (time_step,                'Particle dynamics time steps', '+'),
                          (isolation_threshold_cis,  'Cis contact isolation threshold', '+0'),
                          (isolation_threshold_trans,'Trans contact isolation threshold', '+0'),
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
     
  if temp_steps_by_stage:
    for ts in temp_steps_by_stage:
      if ts <= 0.0:
        critical('temp steps by stage must all be positive')
    if len(temp_steps_by_stage) != len(particle_sizes):
      critical('temp steps by stage must be of same length as particle sizes')
  else:
    temp_steps_by_stage = len(particle_sizes) * [temp_steps]
        
  if dynamics_steps_by_stage:
    for ds in dynamics_steps_by_stage:
      if ds <= 0.0:
        critical('dynamic steps by stage must all be positive')
    if len(dynamics_steps_by_stage) != len(particle_sizes):
      critical('dynamic steps by stage must be of same length as particle sizes')
  else:
    dynamics_steps_by_stage = len(particle_sizes) * [dynamics_steps]
        
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

  anneal_params = {'temp_start':temp_start, 'temp_end':temp_end, 'temp_steps_by_stage':temp_steps_by_stage,
                   'dynamics_steps_by_stage':dynamics_steps_by_stage, 'time_step':time_step}
  
  isolation_threshold_cis *= int(1e6)
  isolation_threshold_trans *= int(1e6)
  struct_ambig_size *= int(1e6)
  remove_models_size *= int(1e6)
  
  calc_genome_structure(ncc_file_path, save_path, general_calc_params, anneal_params,
                        particle_sizes, num_models, isolation_threshold_cis, isolation_threshold_trans,
                        out_format, num_cpu, start_coords_path, save_intermediate, save_intermediate_ncc,
                        have_diploid, struct_ambig_size, remove_models_size)

  if LOG_FILE_OBJ:
    LOG_FILE_OBJ.close()


# TO DO
# -----
# Allow chromosomes to be specified


