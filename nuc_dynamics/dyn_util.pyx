from libc.math cimport exp, abs, sqrt, ceil
from numpy cimport ndarray, double_t, int_t
import numpy, cython, time, gc, random

BOLTZMANN_K = 0.0019872041

#ctypedef int_t     int
#ctypedef double_t double

class NucCythonError(Exception):

    def __init__(self, err):
        Exception.__init__(self, err)


cdef getRepulsionList(ndarray[int, ndim=2] rep_list,
                      ndarray[double, ndim=2] coords,
                      ndarray[double, ndim=3] regions_1,
                      ndarray[double, ndim=4] regions_2,
                      ndarray[int, ndim=1] idx_1,
                      ndarray[int, ndim=2] idx_2,
                      int s1, int s2, int nCoords, int nRepMax,
                      double rep_dist, ndarray[double, ndim=1] radii,
                      double max_radius):

    cdef int i, j, k, i1, i2, j1, j2, n2
    cdef int a, b, a0, b0, a1, b1, a2, b2, p0, p, q0, q
    
    cdef int n    = 0    # Num close pairs
    cdef int n1 = 0    # Num primary regions
    cdef int s3 = s2/s1
    cdef int n_rep_found = 0
    
    cdef double dx, dy, dz, d2
    cdef double d_lim
    cdef double d_lim2
    cdef double rep_dist2 = rep_dist + max_radius

    if nCoords < s2: # No split
        for i in range(nCoords-2):
            for j in range(i+2, nCoords):
                d_lim = rep_dist + radii[i] + radii[j]
 
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
                
                if n < nRepMax:
                    rep_list[n,0] = i
                    rep_list[n,1] = j
                    n += 1
                
                n_rep_found += 1
                
                
    elif nCoords < s1 * s2: # Single split
        # Calc bounding X regions

        n1 = 0 
        for i in range(0, nCoords, s1): 

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
 
                            d_lim = rep_dist + radii[i] + radii[j]
 
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
                            
                            if n < nRepMax:
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
                    a1 = p * s2 + a * s1             # Region start
                    a2 = min(nCoords, a1 + s1)    # Region limit
 
                    if p == q:
                        n2 = a0 # Only compare other ones in same big box
                    else:
                        n2 = 0    # Compare all in other big box
 
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
 
                                d_lim = rep_dist + radii[i] + radii[j]
 
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
                                
                                if n < nRepMax:
                                    rep_list[n,0] = i
                                    rep_list[n,1] = j
                                    n += 1
                                
                                n_rep_found += 1

    return n, n_rep_found


cdef double getTemp(ndarray[double] masses,
                    ndarray[double, ndim=2] veloc,
                    int nCoords): 
    cdef int i
    cdef double kin = 0.0
    
    for i in range(nCoords):
        kin += masses[i] * (veloc[i,0]*veloc[i,0] + veloc[i,1]*veloc[i,1] + veloc[i,2]*veloc[i,2])

    return kin / (3 * nCoords * BOLTZMANN_K)
    

def getStats(ndarray[int, ndim=2] restIndices,
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


cdef void updateMotion(ndarray[double] masses,
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


cdef void updateVelocity(ndarray[double] masses,
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


cdef double getRepulsiveForce(ndarray[int, ndim=2] repList,
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
        

cdef double getRestraintForce(ndarray[double, ndim=2] forces,
                              ndarray[double, ndim=2] coords,
                              ndarray[int, ndim=2] restIndices,
                              ndarray[double, ndim=2] restLimits,
                              ndarray[int, ndim=1] restAmbig,
                              int nRest, double fConst, double exponent=2.0,
                              double distSwitch=0.5, double asymptote=1.0):
                             
    cdef int i, j, k, n, nAmbig
    cdef double a, b, da, d, dmin, dmax, dx, dy, dz
    cdef double r, r2, s2, rjk, ujk, force = 0, t

    b = asymptote*distSwitch*distSwitch - exponent*distSwitch*distSwitch*distSwitch
    a = distSwitch*distSwitch - asymptote*distSwitch - b/ distSwitch

    i = 0
    
    while i < nRest:


        nAmbig = restAmbig[i]
        r2 = 0.0
        
        if nAmbig == 1:
            j = restIndices[i,0]
            k = restIndices[i,1]
            
            if j != k:
                dx = coords[j,0] - coords[k,0]
                dy = coords[j,1] - coords[k,1]
                dz = coords[j,2] - coords[k,2]
                r2 = dx*dx + dy*dy + dz*dz
        
        else:
            for n in range(nAmbig):
                j = restIndices[i+n,0]
                k = restIndices[i+n,1]
                
                if j != k:
                    dx = coords[j,0] - coords[k,0]
                    dy = coords[j,1] - coords[k,1]
                    dz = coords[j,2] - coords[k,2]
                    r = dx*dx + dy*dy + dz*dz
                    r2 += 1.0 / (r * r)

            if r2 > 0:
                r2 = 1.0 / sqrt(r2)

        if r2 <= 0:
            i += nAmbig
            continue

        dmin = restLimits[i,0]
        dmax = restLimits[i,1]

        da = dmax + distSwitch

        if r2 < dmin*dmin:
            r2 = max(r2, 1e-8)
            r = sqrt(r2)
            d = dmin - r
            ujk = fConst * d * d
            rjk = fConst * 2 * d
        
        elif r2 > dmax*dmax:
            r = sqrt(r2)
            d = r - dmax

            if r <= da:
                ujk = fConst * d * d
                rjk = - fConst * 2 * d

            else:
                ujk = fConst * (a + asymptote*d + b/d)
                rjk = - fConst * (asymptote - b/(d*d))

        else:
            ujk = rjk = 0
            r = 1.0

        force += ujk

        if nAmbig == 1:
            j = restIndices[i,0]
            k = restIndices[i,1]

            if j == k:
                i += nAmbig
                continue
            
            t = rjk / r
            dx = coords[j,0] - coords[k,0]
            dy = coords[j,1] - coords[k,1]
            dz = coords[j,2] - coords[k,2]
            
            dx *= t
            dy *= t
            dz *= t
            
            forces[j,0] += dx
            forces[j,1] += dy
            forces[j,2] += dz
            
            forces[k,0] -= dx
            forces[k,1] -= dy
            forces[k,2] -= dz

        else:

            for n in range(nAmbig):
                j = restIndices[i+n,0]
                k = restIndices[i+n,1]
                
                if j == k:
                    continue
                    
                dx = coords[j,0] - coords[k,0]
                dy = coords[j,1] - coords[k,1]
                dz = coords[j,2] - coords[k,2]
                
                s2 = dx*dx + dy*dy + dz*dz
                t = rjk * r2 * r2 * r / (s2 * s2 * s2)
                
                dx *= t
                dy *= t
                dz *= t
                
                forces[j,0] += dx
                forces[k,0] -= dx
                forces[j,1] += dy
                forces[k,1] -= dy
                forces[j,2] += dz
                forces[k,2] -= dz
            
        i += nAmbig

    return force


def getSupportedPairs(ndarray[int, ndim=2] positions,
                      int threshold=2000000,
                      int posErr=100):
                                    
    cdef int i, j, n = len(positions)
    cdef int pA, pB, pC, pD
    cdef ndarray[int, ndim=1] supported = numpy.zeros(n, numpy.int32)
    
    for i in range(n):
        pA = positions[i,0]
        pB = positions[i,1]

        for j in range(n):
            if j == i:
                continue
            
            pC = positions[j,0]
            pD = positions[j,1]
            
            if (posErr < abs(pC-pA) < threshold) and (posErr < abs(pD-pB) < threshold):
                supported[i] = 1
                break
 
            elif (posErr < abs(pD-pA) < threshold) and (posErr < abs(pC-pB) < threshold):
                supported[i] = 1
                break
        
    indices = supported.nonzero()[0]
    
    return indices


def getInterpolatedCoords(chromosomes, posDict, prevPosDict, ndarray[double, ndim=2] coords):
    """
    Interpolate x,y,z particle positions for an array of seq positions to a new    
    seq positions e.g. for a change in binned resolution.
    """
    
    cdef int i, j, i0, j0, p1, p2, n, m, d, dMin
    cdef double f, g
    cdef ndarray[int, ndim=1] positions
    cdef ndarray[int, ndim=1] prevPositions
    
    cdef int a, b
    
    # Get index offsets for each chromo    
    i = 0
    offsets = {}
    for chrA in chromosomes:
        offsets[chrA] = i
        i += len(posDict[chrA])
    
    a = i
    cdef ndarray[double, ndim=2] newCoords = numpy.empty((i, 3), float)
    
    i = 0
    prevOffsets = {}
    for chrA in chromosomes:
        prevOffsets[chrA] = i
        i += len(prevPosDict[chrA])    
    
    b = i
    
    for chrA in posDict:
        i0 = offsets[chrA]
        j0 = prevOffsets[chrA]
        positions = posDict[chrA]
        prevPositions = prevPosDict[chrA]
        n = len(positions)
        m = len(prevPositions)
        
        for i in range(n):

            #find closest old positions for coordinate interpolation
            p1 = 0
            dMin = positions[i]-prevPositions[0]
            
            for j in range(1,m):
                d = positions[i]-prevPositions[j]
                
                if abs(d) < abs(dMin):
                    p1 = j
                    dMin = d #closest pos
                    
                elif abs(d) > abs(dMin): # Seq positions were in order
                    break    
        
            if dMin == 0: #new position coincides with an old position
                p2 = p1
         
            elif dMin > 0: #new pos is above p1
                p2 = min(p1+1, m-1)
            
            else: #new pos is below p1
                p2 = p1
                p1 = max(0, p1-1)
                dMin = positions[i] - prevPositions[p1]
            
            #calculate coordinates
            if prevPositions[p2] == prevPositions[p1]:
                newCoords[i0+i, 0] = coords[j0+p1, 0]
                newCoords[i0+i, 1] = coords[j0+p1, 1]
                newCoords[i0+i, 2] = coords[j0+p1, 2]
 
            else: #interpolate
                f = <float>dMin/<float>(prevPositions[p2]-prevPositions[p1])
                g = 1.0 - f
                
                newCoords[i0+i, 0] = g * coords[j0+p1, 0] + f * coords[j0+p2, 0]
                newCoords[i0+i, 1] = g * coords[j0+p1, 1] + f * coords[j0+p2, 1]
                newCoords[i0+i, 2] = g * coords[j0+p1, 2] + f * coords[j0+p2, 2]
    
    return newCoords


def runDynamics(ndarray[double, ndim=2] coords,
                ndarray[double, ndim=1] masses,
                ndarray[double, ndim=1] radii,
                ndarray[int, ndim=2] restIndices,
                ndarray[double, ndim=2] restLimits,
                ndarray[int, ndim=1] restAmbig,
                double tRef=1000.0, double tStep=0.001, int nSteps=1000,
                double fConstR=1.0, double fConstD=25.0, double beta=10.0,
                double repDist=2.0, # A heuristic to limit the repulsion list
                double tTaken=0.0, int printInterval=10000,
                double tot0=20.458, int nRepMax=0):

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

    if len(restAmbig) != nRest:
        raise NucCythonError('Size of ambiguity list does not match number of restraints')     
    
    cdef int i, j, n, step, nViol, nRep = 0
    
    if not nRepMax:
        nRepMax = nCoords*10 # Dictates initial size of repulsion list, though this is checked at startup
        
    cdef double d2, maxDelta, dx, dy, dz, ek, rmsd, tStep0, temp, fDist, fRep
    cdef double deltaLim = 0.25 * repDist * repDist

    tStep0 = tStep * tot0 
    beta /= tot0

    veloc = numpy.random.normal(0.0, 1.0, (nCoords, 3))
    veloc *= sqrt(tRef / getTemp(masses, veloc, nCoords))

    cdef ndarray[int,        ndim=2] rep_list = numpy.zeros((nRepMax, 2), numpy.int32)    
    cdef ndarray[double, ndim=2] coordsPrev = numpy.empty((nCoords, 3)) # typing this array important for speed 
    cdef ndarray[double, ndim=2] accel = numpy.zeros((nCoords, 3))            # this less so
    cdef ndarray[double, ndim=2] forces = numpy.empty((nCoords, 3))
    
    cdef int s1 = 8                # Small bounding box size
    cdef int s2 = 32 * s1    # Large bounding box size
    cdef int s0
    cdef int n_rep_found
    
    if nCoords >= s2 * s1: # Double split
        s0 = s2
    else:
        s0 = s1
        
    cdef ndarray[double, ndim=3] regions_1 = numpy.zeros((1+nCoords/s0, 3, 2))     # Large bounding boxes
    cdef ndarray[double, ndim=4] regions_2 = numpy.zeros((1+nCoords/s0, s2/s1, 3, 2))     # Small bounding boxes
        
    cdef ndarray[int, ndim=1] idx_1 = numpy.zeros(len(regions_1), numpy.int32)
    cdef ndarray[int, ndim=2] idx_2 = numpy.zeros((len(regions_1), s1), numpy.int32)
    cdef double max_radius = radii.max()
    
    t0 = time.time()

    for step in range(nSteps):

        if step == 0:
         
            nRep, n_rep_found = getRepulsionList(rep_list, coords, regions_1, regions_2, idx_1, idx_2, s1, s2, nCoords, nRepMax, repDist, radii, max_radius)
            
            if n_rep_found > nRepMax:
                nRepMax = numpy.int32(n_rep_found * 1.1)
                rep_list = numpy.zeros((nRepMax, 2), numpy.int32) 
            
            for i in range(nCoords):
                coordsPrev[i,0] = coords[i,0]
                coordsPrev[i,1] = coords[i,1]
                coordsPrev[i,2] = coords[i,2]
                forces[i,0] = 0.0
                forces[i,1] = 0.0
                forces[i,2] = 0.0
             
            fRep = getRepulsiveForce(rep_list, forces, coords, nRep,    fConstR, radii)
            fDist = getRestraintForce(forces, coords, restIndices, restLimits, restAmbig, nRest, fConstD)

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
                    
                    if maxDelta > deltaLim:
                        break                        

            if maxDelta > deltaLim:
                nRep, n_rep_found = getRepulsionList(rep_list, coords, regions_1, regions_2, idx_1, idx_2, s1, s2, nCoords, nRepMax, repDist, radii, max_radius) # Handle errors
                
                if n_rep_found > nRepMax:
                    nRepMax = numpy.int32(n_rep_found * 1.1)
                    rep_list = numpy.zeros((nRepMax, 2), numpy.int32)
                
                for i in range(nCoords):
                    coordsPrev[i,0] = coords[i,0]
                    coordsPrev[i,1] = coords[i,1]
                    coordsPrev[i,2] = coords[i,2]
 
        updateMotion(masses, forces, accel, veloc, coords, nCoords, tRef, tStep0, beta)
        
        for i in range(nCoords):
            forces[i,0] = 0.0
            forces[i,1] = 0.0
            forces[i,2] = 0.0

        fRep    = getRepulsiveForce(rep_list, forces, coords, nRep, fConstR, radii)
        fDist = getRestraintForce(forces, coords, restIndices, restLimits, restAmbig, nRest, fConstD)

        updateVelocity(masses, forces, accel, veloc, nCoords, tRef, tStep0,    beta)

        if (printInterval > 0) and step % printInterval == 0:
            temp = getTemp(masses, veloc, nCoords)
            nViol, rmsd = getStats(restIndices, restLimits, coords, nRest)
            
            data = (temp, fRep, fDist, rmsd, nViol, nRep)
            print('temp:%7.2lf    fRep:%7.2lf    fDist:%7.2lf    rmsd:%7.2lf    nViol:%5d    nRep:%5d' % data)

        tTaken += tStep

    return tTaken, n_rep_found


def calc_restraints(chromosomes, contact_dict, int particle_size=10000,
                    float scale=1.0, float exponent=-0.33,
                    float lower=0.8, float upper=1.2,
                    int min_count=1):
    """
    Function to convert single-cell contact data into distance restraints 
    for structure calculations.
    """

    cdef int i, j, k, a, b, nc, n, na, nb
    cdef double dist
    cdef ndarray[long, ndim=2] contacts            # Contact matrix (3:(posA, posB, nObs), nContacts)
    cdef ndarray[double, ndim=2] restraints    # Distance restraints (6:(), nRestraints)
    cdef ndarray[int, ndim=2] bin_matrix         # Temp array for binned contacts
    cdef ndarray[int, ndim=1] seq_pos_a
    cdef ndarray[int, ndim=1] seq_pos_b
    cdef ndarray[int, ndim=2] limits                 # shape: (chromoId, 2:[start, end])

    num_contacts_dict = {} # Total num contacts for each chromosomes
    pos_dict = {}                    # Start sequence position for each particle in each chromosome
    restraint_dict = {}        # Final restraints for each pair of chromosomes
    
    chromos = set(chromosomes)
    nc = len(chromosomes)
     
    # Num contacts per chromo for mem allocations
    for chrA in contact_dict:
        if chrA not in chromos:
            continue
        
        for chrB in contact_dict[chrA]:
            if chrB not in chromos:
                continue
                
            contacts = contact_dict[chrA][chrB]
            n = len(contacts[0])
            
            if chrA in num_contacts_dict:
                num_contacts_dict[chrA] += n
            else:
                num_contacts_dict[chrA] = n
            
            if chrB in num_contacts_dict:
                num_contacts_dict[chrB] += n
            else:
                num_contacts_dict[chrB] = n
                
    chromo_idx = {chromo:i for i, chromo in enumerate(chromosomes)}
    limits = numpy.zeros((nc,2), numpy.int32)
    
    # Get chromosome ranges
    for chrA in contact_dict:
        if chrA not in chromos:
            continue

        a = chromo_idx[chrA]
        
        for chrB in contact_dict[chrA]:
            if chrB not in chromos:
                continue
                
            b = chromo_idx[chrB]
            contacts = contact_dict[chrA][chrB]
            n = len(contacts[0])
            
            for i in range(n):
                
                if limits[a,0] == 0: # zero is not a valid seq pos anyhow
                    limits[a,0] = contacts[0,i]
                elif contacts[0,i] < limits[a,0]:
                    limits[a,0] = contacts[0,i]
                
                if limits[b,0] == 0:
                    limits[b,0] = contacts[1,i]
                elif contacts[1,i] < limits[b,0]:
                    limits[b,0] = contacts[1,i]
                
                if contacts[0,i] > limits[a,1]:
                    limits[a,1] = contacts[0,i]
            
                if contacts[1,i] > limits[b,1]:
                    limits[b,1] = contacts[1,i]

                        
    # Shift extremities of chromosome ranges to lie exactly on particle region edges
    for a in range(nc):
        limits[a,0] = particle_size * (limits[a,0]/particle_size)
        limits[a,1] = particle_size * <int>(ceil(<float>limits[a,1]/particle_size))

    # Get particle region start positions for each chromosome    
    for chromo in chromosomes:
        a = chromo_idx[chromo]
        pos_dict[chromo] = numpy.arange(limits[a,0], limits[a,1] + particle_size, particle_size, numpy.int32)    
        
    # Get restraint indices, do binning of contact observations
    for chrA in contact_dict:
        if chrA not in chromos:
            continue
            
        a = chromo_idx[chrA]
        na = len(pos_dict[chrA])
        seq_pos_a = pos_dict[chrA]
        restraint_dict[chrA] = {}
        
        for chrB in contact_dict[chrA]:
            if chrB not in chromos:
                continue
                
            b = chromo_idx[chrB]
            nb = len(pos_dict[chrB])
            seq_pos_b = pos_dict[chrB]
            
            contacts = contact_dict[chrA][chrB]
            n = len(contacts[0])
                
            restraints = numpy.empty((6, n), float)
            bin_matrix = numpy.zeros((na, nb), numpy.int32)

            for i in range(n):
                
                # Find bin index for chromo A
                for j in range(na):
                    if seq_pos_a[j] >= contacts[0,i]:
                        break
                
                else:
                    continue
                
                # Find bin index for chromo B
                for k in range(nb):
                    if seq_pos_b[k] >= contacts[1,i]:
                        break
                        
                else:
                    continue
                
                bin_matrix[j,k] += contacts[2,i]            
                
            #loop over all binned contacts, and calculate the constraint target distance
            #using a powerlaw function and the number of observations
            k = 0
            for i in range(na):
                for j in range(nb):
                    if bin_matrix[i,j] > 0:
                        if bin_matrix[i,j] < min_count:
                            continue
                            
                        dist = scale * bin_matrix[i,j] ** exponent
                        
                        restraints[0,k] = <double>i #binA
                        restraints[1,k] = <double>j #binB
                        restraints[2,k] = 1.0 # Weighting not currently used
                        restraints[3,k] = dist # target value
                        restraints[4,k] = dist * lower # constraint lower bound
                        restraints[5,k] = dist * upper # constraint upper bound
                        
                        k += 1

            restraint_dict[chrA][chrB] = restraints[:,:k]
                
    return restraint_dict, pos_dict


def concatenate_restraints(restraint_dict, pos_dict, particle_size,
                           backbone_lower=0.1, backbone_upper=1.1):
    """
    Joins restraints stored in a dict by chromo pairs into long concatenated arrays.
    Indices of restraints relate to concatenated chromo seq pos.
    Add-in all the backbone restraints for sequential particles.
    """
    
    cdef int i, n, num_restraints, m
    cdef int start_a, start_b
    cdef float bbl = numpy.float32(backbone_lower)
    cdef float bbu = numpy.float32(backbone_upper)
    cdef double dist, size = particle_size
    
    # Get total max number restraints and final restraint index offset for all chromos        
    i = 0
    num_restraints = 0
    chromo_idx_offset = {}
    
    for chr_a in sorted(pos_dict):
        n = len(pos_dict[chr_a])
        chromo_idx_offset[chr_a] = i
        num_restraints += n-1 # One fewer because no link at end of chain
        i += n
                
    for chr_a in restraint_dict:
        for chr_b in restraint_dict[chr_a]:
            num_restraints += restraint_dict[chr_a][chr_b].shape[1]
 
    # Loop allocated arrays
    cdef ndarray[int, ndim=1] positions
    cdef ndarray[double, ndim=2] restraints
    
    # Final arrays which will hold identities of restrained particle pairs
    # and the restraint distances for each
    cdef ndarray[int, ndim=2] particle_indices = numpy.empty((num_restraints, 2), numpy.int32)
    cdef ndarray[double, ndim=2] distances = numpy.empty((num_restraints, 2), float)

    # Add backbone path restraints     
    m = 0
    for chr_a in pos_dict:
        
        positions = pos_dict[chr_a]
        n = len(positions)
        start_a = chromo_idx_offset[chr_a]
        
        for i in range(n-1):
            
            # Normally this is 1.0 for regular sized particles
            dist = (positions[i+1]-positions[i])/size
            
            particle_indices[m,0] = i + start_a
            particle_indices[m,1] = i + start_a + 1
            
            distances[m,0] = dist * bbl # lower
            distances[m,1] = dist * bbu # upper
            
            m += 1
            
    # Add regular restraints for chromo pairs 
    for chr_a in restraint_dict:
        start_a = chromo_idx_offset[chr_a] # Offset for chromo A particles
     
        for chr_b in restraint_dict[chr_a]:
            start_b = chromo_idx_offset[chr_b] # Offset for chromo B particles
            
            restraints = restraint_dict[chr_a][chr_b]
            n = restraints.shape[1]
            
            for i in range(n):
                particle_indices[m,0] = <int>restraints[0,i] + start_a
                particle_indices[m,1] = <int>restraints[1,i] + start_b
 
                distances[m,0] = restraints[4,i] # lower
                distances[m,1] = restraints[5,i] # upper
                
                m += 1
 
    return particle_indices, distances
        
