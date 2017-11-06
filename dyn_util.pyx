from libc.math cimport exp, abs, sqrt, ceil, pow
from numpy cimport ndarray, double_t, int_t, dtype
from numpy.math cimport INFINITY
import numpy, time

BOLTZMANN_K = 0.0019872041

#ctypedef int_t   int
#ctypedef double_t double

class NucCythonError(Exception):

  def __init__(self, err):

    Exception.__init__(self, err)


cdef int getRepulsionList(ndarray[int,   ndim=2] repList,
                          ndarray[double, ndim=2] coords,
                          ndarray[double, ndim=1] repDists,
                          ndarray[double, ndim=1] radii,
                          ndarray[double, ndim=1] masses):

  cdef int i, j
  cdef int n = 0
  cdef double dx, dy, dz, d2
  cdef double distLim
  cdef double distLim2

  for i in range(len(coords)-2):
    if masses[i] == INFINITY:
      continue

    for j in range(i+2, len(coords)):
      if masses[j] == INFINITY:
        continue

      distLim = repDists[i] + radii[i] + repDists[j] + radii[j]
      distLim2 = distLim * distLim

      dx = coords[i,0] - coords[j,0]
      if abs(dx) > distLim:
        continue

      dy = coords[i,1] - coords[j,1]
      if abs(dy) > distLim:
        continue

      dz = coords[i,2] - coords[j,2]
      if abs(dz) > distLim:
        continue

      d2 = dx*dx + dy*dy + dz*dz

      if d2 > distLim2:
        continue

      # If max is exceeded, array will be resized and recalculated
      if n < len(repList):
        repList[n,0] = i
        repList[n,1] = j

      n += 1

  return n


cpdef double getTemp(ndarray[double, ndim=1] masses,
                     ndarray[double, ndim=2] veloc,
                     int nCoords):
  cdef int i
  cdef double kin = 0.0

  for i in range(nCoords):
    if masses[i] == INFINITY:
      continue
    kin += masses[i] * (veloc[i,0]*veloc[i,0] + veloc[i,1]*veloc[i,1] + veloc[i,2]*veloc[i,2])

  return kin / (3 * nCoords * BOLTZMANN_K)


def getStats(ndarray[int,   ndim=2] restIndices,
             ndarray[double, ndim=2] restLimits,
             ndarray[double, ndim=2] coords,
             int nRest):

  cdef int i, nViol = 0
  cdef int j, k
  cdef double viol, dmin, dmax, dx, dy, dz, r, s = 0

  for i in range(nRest):
    j = restIndices[i,0]
    k = restIndices[i,1]

    if j == k:
      continue

    dmin = restLimits[i,0]
    dmax = restLimits[i,1]

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


cdef double getRepulsiveForce(ndarray[int,   ndim=2] repList,
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
    j = repList[i,0]
    k = repList[i,1]
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


cpdef double getRestraintForce(ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              ndarray[int,   ndim=2] restIndices,
                              ndarray[double, ndim=2] restLimits,
                              ndarray[double, ndim=1] restWeight,
                              ndarray[int, ndim=1] restAmbig,
                              double fConst, double exponent=2.0,
                              double switchRatio=0.5, double asymptote=1.0):

  cdef int i, j, k, n, m, nAmbig
  cdef double a, b, d, dmin, dmax, dx, dy, dz, distSwitch
  cdef double r, r2, s2, rjk, ujk, force = 0, t

  for m in range(len(restAmbig) - 1):
    nAmbig = restAmbig[m+1] - restAmbig[m]
    i = restAmbig[m]
    r2 = 0.0

    for n in range(nAmbig):
      j = restIndices[i+n,0]
      k = restIndices[i+n,1]

      if j == k:
        continue

      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]
      r = dx*dx + dy*dy + dz*dz
      r2 += 1.0 / (r * r)

    if r2 <= 0:
      continue

    r2 = 1.0 / sqrt(r2)

    dmin = restLimits[i,0]
    dmax = restLimits[i,1]
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
      j = restIndices[i+n,0]
      k = restIndices[i+n,1]

      if j == k:
        continue

      dx = coords[j,0] - coords[k,0]
      dy = coords[j,1] - coords[k,1]
      dz = coords[j,2] - coords[k,2]

      s2 = max(dx*dx + dy*dy + dz*dz, 1e-08)
      t = rjk * pow(r2, 2.5) / (s2 * s2 * s2) * restWeight[i+n]

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


def runDynamics(ndarray[double, ndim=2] coords,
                ndarray[double, ndim=1] masses,
                ndarray[double, ndim=1] radii,
                ndarray[double, ndim=1] repDists,
                ndarray[int, ndim=2] restIndices,
                ndarray[double, ndim=2] restLimits,
                ndarray[double, ndim=1] restWeight,
                ndarray[int, ndim=1] restAmbig,
                double tRef=1000.0, double tStep=0.001, int nSteps=1000,
                double fConstR=1.0, double fConstD=25.0, double beta=10.0,
                double tTaken=0.0, int printInterval=10000,
                double tot0=20.458):

  cdef int nRest = len(restIndices)
  cdef int nCoords = len(coords)

  if nCoords < 2:
    raise NucCythonError('Too few coodinates')

  indices = set(restIndices.ravel())
  if min(indices) < 0:
    raise NucCythonError('Restraint index negative')

  if max(indices) >= nCoords:
    data = (max(indices), nCoords)
    raise NucCythonError('Restraint index "%d" out of bounds (> %d)' % data)

  if nCoords != len(masses):
    raise NucCythonError('Masses list size does not match coordinates')

  if nRest != len(restLimits):
    raise NucCythonError('Number of restraint index pairs does not match number of restraint limits')

  cdef int i, j, n, step, nViol, nRep = 0

  cdef double d2, dx, dy, dz, ek, rmsd, tStep0, temp, fDist, fRep
  cdef ndarray[double, ndim=1] deltaLim = repDists * repDists
  cdef double Langevin_gamma

  tStep0 = tStep * tot0
  beta /= tot0

  cdef ndarray[double, ndim=2] veloc = numpy.random.normal(0.0, 1.0, (nCoords, 3))
  veloc *= sqrt(tRef / getTemp(masses, veloc, nCoords))
  for i, m in enumerate(masses):
    if m == INFINITY:
      veloc[i] = 0

  cdef ndarray[int, ndim=2] repList = numpy.empty((0, 2), numpy.int32)
  cdef ndarray[double, ndim=2] coordsPrev = numpy.array(coords)
  cdef ndarray[double, ndim=2] accel = numpy.zeros((nCoords, 3))
  cdef ndarray[double, ndim=2] forces = numpy.zeros((nCoords, 3))

  cdef double t0 = time.time()

  nRep = getRepulsionList(repList, coords, repDists, radii, masses)
  # Allocate with some padding
  repList = numpy.resize(repList, (int(nRep * 1.2), 2))
  nRep = getRepulsionList(repList, coords, repDists, radii, masses)

  fRep = getRepulsiveForce(repList, forces, coords, nRep,  fConstR, radii)
  fDist = getRestraintForce(forces, coords, restIndices, restLimits,
                            restWeight, restAmbig, fConstD)

  for step in range(nSteps):
    for i in range(nCoords):
      dx = coords[i,0] - coordsPrev[i,0]
      dy = coords[i,1] - coordsPrev[i,1]
      dz = coords[i,2] - coordsPrev[i,2]
      if dx*dx + dy*dy + dz*dz > deltaLim[i]:
        nRep = getRepulsionList(repList, coords, repDists, radii, masses)
        if nRep > len(repList):
          repList = numpy.resize(repList, (int(nRep * 1.2), 2))
          nRep = getRepulsionList(repList, coords, repDists, radii, masses)
        elif nRep < (len(repList) // 2):
          repList = numpy.resize(repList, (int(nRep * 1.2), 2))

        for i in range(nCoords):
          coordsPrev[i,0] = coords[i,0]
          coordsPrev[i,1] = coords[i,1]
          coordsPrev[i,2] = coords[i,2]
        break # Already re-calculated, no need to check more

    updateMotion(masses, forces, accel, veloc, coords, nCoords, tRef, tStep0, beta)

    for i in range(nCoords):
      forces[i,0] = 0.0
      forces[i,1] = 0.0
      forces[i,2] = 0.0

    fRep  = getRepulsiveForce(repList, forces, coords, nRep, fConstR, radii)
    fDist = getRestraintForce(forces, coords, restIndices, restLimits,
                              restWeight, restAmbig, fConstD)

    updateVelocity(masses, forces, accel, veloc, nCoords, tRef, tStep0,  beta)

    if (printInterval > 0) and step % printInterval == 0:
      temp = getTemp(masses, veloc, nCoords)
      nViol, rmsd = getStats(restIndices, restLimits, coords, nRest)

      data = (temp, fRep, fDist, rmsd, nViol, nRep)
      print('temp:%7.2lf  fRep:%7.2lf  fDist:%7.2lf  rmsd:%7.2lf  nViol:%5d  nRep:%5d' % data)

    tTaken += tStep

  return tTaken