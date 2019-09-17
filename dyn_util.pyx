from libc.math cimport exp, abs, sqrt, ceil, pow
from numpy cimport ndarray, double_t, int_t, dtype
from numpy.math cimport INFINITY
import numpy, time, sys

BOLTZMANN_K = 0.0019872041

#ctypedef int_t   int
#ctypedef double_t double

class NucCythonError(Exception):

  def __init__(self, err):

    Exception.__init__(self, err)


cdef getRepulsionList(ndarray[long,   ndim=2] rep_list,
                      ndarray[double, ndim=2] coords,
                      ndarray[double, ndim=3] regions_1,
                      ndarray[double, ndim=4] regions_2,
                      ndarray[int, ndim=1] idx_1,
                      ndarray[int, ndim=2] idx_2,
                      int s1, int s2, int nCoords, int n_rep_max,
                      #double rep_dist, ndarray[double, ndim=1] radii,
                      ndarray[double, ndim=1] rep_dists, ndarray[double, ndim=1] radii,
                      double max_radius):
  
  cdef int i, j, k, i1, i2, j1, j2, n2
  cdef int a, b, a0, b0, a1, b1, a2, b2, p0, p, q0, q
  
  cdef int n  = 0  # Num close pairs
  cdef int n1 = 0  # Num primary regions
  cdef int s3 = s2/s1
  cdef int n_rep_found = 0
  
  cdef double dx, dy, dz, d2
  cdef double d_lim
  cdef double d_lim2
  #cdef double rep_dist2 = rep_dist + max_radius
  cdef double rep_dist2

  if nCoords < s2: # No split
    for i in range(nCoords-2):
      for j in range(i+2, nCoords):
        #d_lim = rep_dist + radii[i] + radii[j]
        d_lim = rep_dists[i] + radii[i] + rep_dists[j] + radii[j]
 
        dx = coords[i,0] - coords[j,0]
        if abs(dx) > d_lim:
          continue

        dy = coords[i,1] - coords[j,1]
        if abs(dy) > d_lim:
          continue

        dz = coords[i,2] - coords[j,2]
        if abs(dz) > d_lim:
          continue

        d2 = dx*dx + dy*dy + dz*dz
 
        d_lim2 = d_lim * d_lim
        if d2 > d_lim2:
          continue
        
        if n < n_rep_max:
          rep_list[n,0] = i
          rep_list[n,1] = j
          n += 1
        
        n_rep_found += 1
        
  elif nCoords < s1 * s2: # Single split
    # Calc bounding X regions

    n1 = 0 
    for i in range(0, nCoords, s1): 

      rep_dist2 = 2*rep_dists[i] + max_radius
      
      for k in range(3):
        regions_1[n1,k,0] = coords[i,k]
        regions_1[n1,k,1] = coords[i,k]
        
      for j in range(i+1, i+s1):
        if j >= nCoords:
          break
      
        if coords[j,0] < regions_1[n1,0,0]:
          regions_1[n1,0,0] = coords[j,0]
 
        if coords[j,0] > regions_1[n1,0,1]:
          regions_1[n1,0,1] = coords[j,0]

        if coords[j,1] < regions_1[n1,1,0]:
          regions_1[n1,1,0] = coords[j,1]
 
        if coords[j,1] > regions_1[n1,1,1]:
          regions_1[n1,1,1] = coords[j,1]
          
        if coords[j,2] < regions_1[n1,2,0]:
          regions_1[n1,2,0] = coords[j,2]
 
        if coords[j,2] > regions_1[n1,2,1]:
          regions_1[n1,2,1] = coords[j,2]
      
      for k in range(3):
        regions_1[n1,k,0] -= rep_dist2
        regions_1[n1,k,1] += rep_dist2
     
      n1 += 1 # Number of primary regions     
      
    idx_1 = regions_1[:,0,0].argsort(kind='heapsort').astype(numpy.int32) # Sort X starts
    
    for a0 in range(n1):
      a = idx_1[a0]
      a1 = a * s1
      a2 = min(nCoords, a1 + s1)
            
      # Compare between regions 
      for b0 in range(a0, n1):
        b = idx_1[b0]
        b1 = b * s1
        b2 = min(nCoords, b1 + s1)
                 
        if regions_1[b,0,0] < regions_1[a,0,1]:

          if regions_1[b,1,1] < regions_1[a,1,0]:
            continue
          
          if regions_1[b,1,0] > regions_1[a,1,1]:
            continue
          
          if regions_1[b,2,1] < regions_1[a,2,0]:
            continue
          
          if regions_1[b,2,0] > regions_1[a,2,1]:
            continue
 
          if a1 < b1:
            i1 = a1
            i2 = a2
            j1 = b1
            j2 = b2
 
          else:
            i1 = b1
            i2 = b2
            j1 = a1
            j2 = a2
          
          for i in range(i1, i2):
            for j in range(j1, j2):
              if j < i+2: # not sequential
                continue
 
              #d_lim = rep_dist + radii[i] + radii[j]
              d_lim = rep_dists[i] + radii[i] + rep_dists[j] + radii[j]
 
              dx = coords[i,0] - coords[j,0]
              if abs(dx) > d_lim:
                continue

              dy = coords[i,1] - coords[j,1]
              if abs(dy) > d_lim:
                continue

              dz = coords[i,2] - coords[j,2]
              if abs(dz) > d_lim:
                continue

              d2 = dx*dx + dy*dy + dz*dz
 
              d_lim2 = d_lim * d_lim
              if d2 > d_lim2:
                continue
              
              if n < n_rep_max:
                rep_list[n,0] = i
                rep_list[n,1] = j
                n += 1
                
              n_rep_found += 1
          
        else: # All subsequent regions do not overlap
          break  

  else: # Double split
    # Calc bounding X regions
    
    n1 = 0
    for i in range(0, nCoords, s2): # Large region starts
    
      rep_dist2 = 2*rep_dists[i] + max_radius
      
      for a in range(3):
        regions_1[n1,a,0] = coords[i,a]
        regions_1[n1,a,1] = coords[i,a]
      
      j = i
      for n2 in range(s3): # Small regions
        for a in range(3):
          regions_2[n1,n2,a,0] = coords[j,a]
          regions_2[n1,n2,a,1] = coords[j,a]
        
        # Bounding box for small region
        
        for k in range(j+1, j+s1): # All remaining points
          if k >= nCoords:
            break
          
          for a in range(3):
            if coords[k,a] < regions_2[n1,n2,a,0]:
              regions_2[n1,n2,a,0] = coords[k,a]
 
            if coords[k,a] > regions_2[n1,n2,a,1]:
              regions_2[n1,n2,a,1] = coords[k,a] 
        
        # Bounding box of large region using small regions
        
        for a in range(3):
          if regions_2[n1,n2,a,0] < regions_1[n1,a,0]:
            regions_1[n1,a,0] = regions_2[n1,n2,a,0]

          if regions_2[n1,n2,a,1] > regions_1[n1,a,1]:
            regions_1[n1,a,1] = regions_2[n1,n2,a,1]

        for a in range(3):
          regions_2[n1,n2,a,0] -= rep_dist2
          regions_2[n1,n2,a,1] += rep_dist2
         
        j += s1
        
        if j >= nCoords:
          break
      
      for a in range(3):
        regions_1[n1,a,0] -= rep_dist2
        regions_1[n1,a,1] += rep_dist2
        
      n1 += 1
     
    idx_1 = regions_1[:,0,0].argsort(kind='heapsort').astype(numpy.int32) # Order X starts
    idx_2 = regions_2[:,:,0,0].argsort(kind='heapsort', axis=1).astype(numpy.int32)    
        
    for p0 in range(n1):
      p = idx_1[p0]
      
      for q0 in range(p0, n1):
        q = idx_1[q0]
                  
        if regions_1[q,0,0] >= regions_1[p,0,1]: # Remainder do not overlap as they are in X order
          break
          
        if regions_1[q,1,1] < regions_1[p,1,0]:
          continue
 
        if regions_1[q,1,0] > regions_1[p,1,1]:
          continue
 
        if regions_1[q,2,1] < regions_1[p,2,0]:
          continue
 
        if regions_1[q,2,0] > regions_1[p,2,1]:
          continue
 
        # Big bounding boxes overlap
 
        for a0 in range(s3):
          
          a = idx_2[p,a0]
          a1 = p * s2 + a * s1       # Region start
          a2 = min(nCoords, a1 + s1)  # Region limit
 
          if p == q:
            n2 = a0 # Only compare other ones in same big box
          else:
            n2 = 0  # Compare all in other big box
 
          # Compare between regions
          for b0 in range(n2, s3):
            b = idx_2[q,b0]
                 
            if regions_2[q,b,0,0] >= regions_2[p,a,0,1]:
              break

            if regions_2[q,b,0,1] < regions_2[p,a,0,0]: # Can happen across different big boxes
              continue
 
            if regions_2[q,b,1,1] < regions_2[p,a,1,0]:
              continue
 
            if regions_2[q,b,1,0] > regions_2[p,a,1,1]:
              continue
 
            if regions_2[q,b,2,1] < regions_2[p,a,2,0]:
              continue
 
            if regions_2[q,b,2,0] > regions_2[p,a,2,1]:
              continue

            b1 = q * s2 + b * s1
            b2 = min(nCoords, b1 + s1)

            if a1 < b1:
              i1 = a1
              i2 = a2
              j1 = b1
              j2 = b2
 
            else:
              i1 = b1
              i2 = b2
              j1 = a1
              j2 = a2
 
            # Small bounding boxes overlap
 
            for i in range(i1, i2):
              for j in range(j1, j2):
                if j < i+2: # avoid sequential
                  continue
 
                #d_lim = rep_dist + radii[i] + radii[j]
                d_lim = rep_dists[i] + radii[i] + rep_dists[j] + radii[j]
 
                dx = coords[i,0] - coords[j,0]
                if abs(dx) > d_lim:
                  continue

                dy = coords[i,1] - coords[j,1]
                if abs(dy) > d_lim:
                  continue

                dz = coords[i,2] - coords[j,2]
                if abs(dz) > d_lim:
                  continue

                d2 = dx*dx + dy*dy + dz*dz
 
                d_lim2 = d_lim * d_lim
                if d2 > d_lim2:
                  continue
                
                if n < n_rep_max:
                  rep_list[n,0] = i
                  rep_list[n,1] = j
                  n += 1
                
                n_rep_found += 1
                
  return n, n_rep_found


cdef double getTemp(ndarray[double, ndim=1] masses,
                    ndarray[double, ndim=2] veloc,
                    int nCoords):
  cdef int i
  cdef double kin = 0.0

  for i in range(nCoords):
    if masses[i] == INFINITY:
      continue

    kin += masses[i] * (veloc[i,0]*veloc[i,0] + veloc[i,1]*veloc[i,1] + veloc[i,2]*veloc[i,2])

  return kin / (3 * nCoords * BOLTZMANN_K)


def getStats(ndarray[int,   ndim=2] rest_indices,
             ndarray[double, ndim=2] rest_limits,
             ndarray[double, ndim=2] coords,
             int nRest):

  cdef int i, nViol = 0
  cdef int j, k
  cdef double viol, dmin, dmax, dx, dy, dz, r, s = 0

  for i in range(nRest):
    j = rest_indices[i,0]
    k = rest_indices[i,1]

    if j == k:
      continue

    dmin = rest_limits[i,0]
    dmax = rest_limits[i,1]

    dx = coords[j,0] - coords[k,0]
    dy = coords[j,1] - coords[k,1]
    dz = coords[j,2] - coords[k,2]
    r = sqrt(dx*dx + dy*dy + dz*dz)

    if r < dmin:
      viol = dmin - r
      nViol += 1

    elif r > dmax:
      viol = r - dmax
      nViol += 1

    else:
      viol = 0

    s += viol * viol

  return nViol, sqrt(s/nRest)


cdef void updateMotion(ndarray[double, ndim=1] masses,
                       ndarray[double, ndim=2] forces,
                       ndarray[double, ndim=2] accel,
                       ndarray[double, ndim=2] veloc,
                       ndarray[double, ndim=2] coords,
                       int nCoords, double tRef,
                       double tStep, double beta):

  cdef int i
  cdef double r, rtStep, temp

  rtStep = 0.5 * tStep * tStep
  temp = getTemp(masses, veloc, nCoords)
  temp = max(temp, 0.001)
  r = beta * (tRef/temp-1.0)

  for i in range(nCoords):

    accel[i,0] = forces[i,0] / masses[i] + r * veloc[i,0]
    accel[i,1] = forces[i,1] / masses[i] + r * veloc[i,1]
    accel[i,2] = forces[i,2] / masses[i] + r * veloc[i,2]

    coords[i,0] += tStep * veloc[i,0] + rtStep * accel[i,0]
    coords[i,1] += tStep * veloc[i,1] + rtStep * accel[i,1]
    coords[i,2] += tStep * veloc[i,2] + rtStep * accel[i,2]

    veloc[i,0] += tStep * accel[i,0]
    veloc[i,1] += tStep * accel[i,1]
    veloc[i,2] += tStep * accel[i,2]


cdef void updateVelocity(ndarray[double, ndim=1] masses,
                         ndarray[double, ndim=2] forces,
                         ndarray[double, ndim=2] accel,
                         ndarray[double, ndim=2] veloc,
                         int nCoords, double tRef,
                         double tStep, double beta):

  cdef int i
  cdef double r, temp

  temp = getTemp(masses, veloc, nCoords)
  #avoid division by 0 temperature
  temp = max(temp, 0.001)
  r = beta * (tRef/temp-1.0)

  for i in range(nCoords):
    veloc[i,0] += 0.5 * tStep * (forces[i,0] / masses[i] + r * veloc[i,0] - accel[i,0])
    veloc[i,1] += 0.5 * tStep * (forces[i,1] / masses[i] + r * veloc[i,1] - accel[i,1])
    veloc[i,2] += 0.5 * tStep * (forces[i,2] / masses[i] + r * veloc[i,2] - accel[i,2])


cdef double getRepulsiveForce(ndarray[long,   ndim=2] rep_list,
                              ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              int nRep, double fConst,
                              ndarray[double, ndim=1] radii):

  cdef int i, j, k
  cdef double dx, dy, dz, d2, dr, rjk
  cdef double force = 0
  cdef double repDist2

  if fConst == 0:
    return force

  for i from 0 <= i < nRep:
    j = rep_list[i,0]
    k = rep_list[i,1]
    repDist = radii[j] + radii[k]
    repDist2 = repDist * repDist


    dx = coords[k,0] - coords[j,0]
    if abs(dx) > repDist:
      continue

    dy = coords[k,1] - coords[j,1]
    if abs(dy) > repDist:
      continue

    dz = coords[k,2] - coords[j,2]
    if abs(dz) > repDist:
      continue

    d2 = dx*dx + dy*dy + dz*dz
    if d2 > repDist2:
      continue

    dr = repDist2 - d2
    #energy contribution
    force += fConst * dr * dr
    rjk = 4 * fConst * dr

    dx *= rjk
    dy *= rjk
    dz *= rjk

    #force contributions
    forces[j,0] -= dx
    forces[k,0] += dx

    forces[j,1] -= dy
    forces[k,1] += dy

    forces[j,2] -= dz
    forces[k,2] += dz

  return force


cdef double getRestraintForce(ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              ndarray[int,   ndim=2] rest_indices,
                              ndarray[double, ndim=2] rest_limits,
                              ndarray[double, ndim=1] rest_weight,
                              ndarray[int, ndim=1] rest_ambig,
                              double fConst, double exponent=2.0,
                              double switchRatio=0.5, double asymptote=1.0):

  cdef int i, j, k, n, m, nAmbig
  cdef double a, b, d, dmin, dmax, dx, dy, dz, distSwitch
  cdef double r, r2, s2, rjk, ujk, force = 0, t

  for m in range(len(rest_ambig) - 1):
    nAmbig = rest_ambig[m+1] - rest_ambig[m]
    i = rest_ambig[m]
    r2 = 0.0

    for n in range(nAmbig):
      j = rest_indices[i+n,0]
      k = rest_indices[i+n,1]

      if j == k:
        continue

      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]
      r = max(dx*dx + dy*dy + dz*dz, 1e-08)
      r2 += 1.0 / (r * r)

    if r2 <= 0:
      continue

    r2 = 1.0 / sqrt(r2)

    dmin = rest_limits[i,0]
    dmax = rest_limits[i,1]
    distSwitch = dmax * switchRatio

    if r2 < dmin*dmin:
      r2 = max(r2, 1e-8)
      d = dmin - sqrt(r2)
      ujk = fConst * d * d
      rjk = fConst * exponent * d

    elif dmin*dmin <= r2 <= dmax*dmax:
      ujk = rjk = 0
      r = 1.0

    elif dmax*dmax < r2 <= (dmax+distSwitch) * (dmax+distSwitch):
      d = sqrt(r2) - dmax
      ujk = fConst * d * d
      rjk = - fConst * exponent * d

    else: # (dmax+distSwitch) ** 2 < r2
      b = distSwitch * distSwitch * distSwitch * exponent * (asymptote - 1)
      a = distSwitch * distSwitch * (1 - 2*asymptote*exponent + exponent)

      d = sqrt(r2) - dmax
      ujk = fConst * (a + asymptote*distSwitch*exponent*d + b/d)
      rjk = - fConst * (asymptote*distSwitch*exponent - b/(d*d))

    force += ujk

    for n in range(nAmbig):
      j = rest_indices[i+n,0]
      k = rest_indices[i+n,1]

      if j == k:
        continue

      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]

      s2 = max(dx*dx + dy*dy + dz*dz, 1e-08)
      t = rjk * pow(r2, 2.5) / (s2 * s2 * s2) * rest_weight[i+n]

      dx *= t
      dy *= t
      dz *= t

      forces[j,0] += dx
      forces[k,0] -= dx
      forces[j,1] += dy
      forces[k,1] -= dy
      forces[j,2] += dz
      forces[k,2] -= dz

  return force


def run_dynamics(ndarray[double, ndim=2] coords,
                 ndarray[double, ndim=1] masses,
                 ndarray[double, ndim=1] radii,
                 ndarray[double, ndim=1] rep_dists,
                 ndarray[int, ndim=2] rest_indices,
                 ndarray[double, ndim=2] rest_limits,
                 ndarray[double, ndim=1] rest_weight,
                 ndarray[int, ndim=1] rest_ambig,
                 double tRef=1000.0, double tStep=0.001, int nSteps=1000,
                 double fConstR=1.0, double fConstD=25.0,
                 double bead_size=1.0, int n_rep_max=0,
                 double beta=10.0, double time_taken=0.0,
                 int print_interval=10000, double tot0=20.458):

  cdef int nRest = len(rest_indices)
  cdef int nCoords = len(coords)
  cdef int i, j, n, step, nViol, nRep = 0
  cdef int s1 = 8        # Small bounding box size
  cdef int s2 = 32 * s1  # Large bounding box size
  cdef int s0, n_rep_found

  if nCoords >= s2 * s1: # Double split
    s0 = s2
  else:
    s0 = s1
  
  if n_rep_max == 0:
    n_rep_max = nCoords * 10

  cdef double d2, dx, dy, dz, ek, rmsd, tStep0, temp, fDist, fRep
  cdef double t0 = time.time()
  cdef double max_radius = radii.max()

  cdef ndarray[double, ndim=1] delta_lim = rep_dists * rep_dists
  cdef ndarray[double, ndim=2] veloc = numpy.random.normal(0.0, 1.0, (nCoords, 3))
  cdef ndarray[double, ndim=2] coordsPrev = numpy.array(coords)
  cdef ndarray[double, ndim=2] accel = numpy.zeros((nCoords, 3))
  cdef ndarray[double, ndim=2] forces = numpy.zeros((nCoords, 3))
  cdef ndarray[double, ndim=3] regions_1 = numpy.zeros((1+nCoords/s0, 3, 2))   # Large bounding boxes
  cdef ndarray[double, ndim=4] regions_2 = numpy.zeros((1+nCoords/s0, s2/s1, 3, 2))   # Small bounding boxes

  cdef ndarray[int, ndim=1] idx_1 = numpy.zeros(len(regions_1), numpy.int32)
  cdef ndarray[int, ndim=2] idx_2 = numpy.zeros((len(regions_1), s1), numpy.int32)
  cdef ndarray[long, ndim=2] rep_list = numpy.empty((n_rep_max, 2), numpy.int64)
   
  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

  indices = set(rest_indices.ravel())
  if min(indices) < 0:
    raise NucCythonError('Restraint index negative')

  if max(indices) >= nCoords:
    data = (max(indices), nCoords)
    raise NucCythonError('Restraint index "%d" out of bounds (> %d)' % data)

  if nCoords != len(masses):
    raise NucCythonError('Masses list size does not match coordinates')

  if nRest != len(rest_limits):
    raise NucCythonError('Number of restraint index pairs does not match number of restraint limits')
  
  tStep0 = tStep * tot0
  beta /= tot0
  
  veloc *= sqrt(tRef / getTemp(masses, veloc, nCoords))
  for i, m in enumerate(masses):
    if m == INFINITY:
      veloc[i] = 0

  for step in range(nSteps):

    if step == 0:
      nRep, n_rep_found = getRepulsionList(rep_list, coords, regions_1, regions_2, idx_1, idx_2,
                                           s1, s2, nCoords, n_rep_max, rep_dists, radii, max_radius)

      if n_rep_found > n_rep_max:
        #print "Adjust A", nRep, n_rep_found, n_rep_max
        n_rep_max = numpy.int32(min(n_rep_found * 1.2, 100000000))
        rep_list = numpy.zeros((n_rep_max, 2), numpy.int64)
        
        nRep, n_rep_found = getRepulsionList(rep_list, coords, regions_1, regions_2, idx_1, idx_2,
                                             s1, s2, nCoords, n_rep_max, rep_dists, radii, max_radius)

      for i in range(nCoords):
        coordsPrev[i,0] = coords[i,0]
        coordsPrev[i,1] = coords[i,1]
        coordsPrev[i,2] = coords[i,2]
        forces[i,0] = 0.0
        forces[i,1] = 0.0
        forces[i,2] = 0.0

      fRep = getRepulsiveForce(rep_list, forces, coords, nRep,  fConstR, radii)
      fDist = getRestraintForce(forces, coords, rest_indices, rest_limits, rest_weight, rest_ambig, fConstD)

      for i in range(nCoords):
        accel[i,0] = forces[i,0] / masses[i]
        accel[i,1] = forces[i,1] / masses[i]
        accel[i,2] = forces[i,2] / masses[i]

    else:
      maxDelta = 0.0
      for i in range(nCoords): 
        dx = coords[i,0] - coordsPrev[i,0]
        dy = coords[i,1] - coordsPrev[i,1]
        dz = coords[i,2] - coordsPrev[i,2]
        d2 = dx*dx + dy*dy + dz*dz

        if d2 > maxDelta:
          maxDelta = d2
  
          if maxDelta > delta_lim[i]:
            break            

      if maxDelta > delta_lim.min():
        nRep, n_rep_found = getRepulsionList(rep_list, coords, regions_1, regions_2, idx_1, idx_2,
                                             s1, s2, nCoords, n_rep_max, rep_dists, radii, max_radius) # Handle errors

        if n_rep_found > n_rep_max:
          #print "Adjust B", nRep, n_rep_found, n_rep_max
          n_rep_max = numpy.int32(min(n_rep_found * 1.2, 100000000))
          rep_list = numpy.zeros((n_rep_max, 2), numpy.int64)

        for i in range(nCoords):
          coordsPrev[i,0] = coords[i,0]
          coordsPrev[i,1] = coords[i,1]
          coordsPrev[i,2] = coords[i,2]

    updateMotion(masses, forces, accel, veloc, coords, nCoords, tRef, tStep0, beta)

    for i in range(nCoords):
      forces[i,0] = 0.0
      forces[i,1] = 0.0
      forces[i,2] = 0.0

    fRep  = getRepulsiveForce(rep_list, forces, coords, nRep, fConstR, radii)
    fDist = getRestraintForce(forces, coords, rest_indices, rest_limits,
                              rest_weight, rest_ambig, fConstD)

    updateVelocity(masses, forces, accel, veloc, nCoords, tRef, tStep0,  beta)

    if (print_interval > 0) and step % print_interval == 0:
      temp = getTemp(masses, veloc, nCoords)
      nViol, rmsd = getStats(rest_indices, rest_limits, coords, nRest)
      
      data = (tRef/(bead_size*bead_size), temp/(bead_size*bead_size), fRep, fDist, rmsd, nViol, nRep, time.time()-t0)
      print('ttemp:%5d temp:%5d fRep:%7.2e fDist:%7.2e rmsd:%7.2lf nViol:%5d nRep:%7d etime:%5.2f' % data)
      sys.stdout.flush()
      
    time_taken += tStep

  return time_taken, n_rep_found
