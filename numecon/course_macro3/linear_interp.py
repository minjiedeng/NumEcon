import numpy as np
import numba
from types import SimpleNamespace

#########
# setup #
#########

def create_interpolator_dict(grids,values,Nxi):

    s = SimpleNamespace()

    # a. grids
    s.x = np.hstack(grids)
    s.dimx = np.int32(len(grids))
    s.Nx = np.array([len(grids[i]) for i in range(s.dimx)],dtype=np.int32)

    # b. values
    s.y = np.vstack(values).ravel()
    s.dimy = np.int32(len(values))
    s.Ny = np.int32(values[0].size)

    # c. number of interpolation points
    s.Nxi = Nxi
    
    # d. create if feasible
    for i in range(s.dimy):
        assert np.all(values[i].shape == s.Nx)

    return s

def create_interpolator(grids,values,Nxi):
    
    s = create_interpolator_dict(grids,values,Nxi)
    return interpolator(s.x,s.dimx,s.Nx,s.y,s.dimy,s.Ny,s.Nxi)

########
# main #
########

spec = [
    ('add',numba.int32[:]),
    ('cube',numba.double[:]),
    ('dimx',numba.int32),
    ('dimy',numba.int32),
    ('facs',numba.int32[:]),
    ('indexes',numba.int32[:]),
    ('ncube',numba.int32),
    ('Nx',numba.int32[:]),
    ('Nxi',numba.int32),
    ('Ny',numba.int32),
    ('pos_left',numba.int32[:]),
    ('pos_left_vec',numba.int32[:]),
    ('reldiff',numba.double[:]),
    ('reldiff_vec',numba.double[:]),
    ('weights',numba.double[:]),
    ('x',numba.double[:]),
    ('y',numba.double[:]),
    ('yi',numba.double[:]),
    ('yi_temp',numba.double[:])
]

@numba.jitclass(spec)
class interpolator():
    
    # setup
    def __init__(self,x,dimx,Nx,y,dimy,Ny,Nxi):

        # a. input
        self.x = x
        self.dimx = dimx
        self.Nx = Nx
        
        self.y = y
        self.dimy = dimy
        self.Ny = Ny

        self.Nxi = Nxi

        # b. calculations
        self.ncube = np.int32(2**(self.dimx-1))
            
        # c. containers
        self.add = np.zeros(self.ncube*self.dimx,dtype=np.int32)
        self.facs = np.zeros(self.dimx,dtype=np.int32)
        self.indexes = np.zeros(self.ncube,dtype=np.int32)
        self.pos_left = np.zeros(self.dimx,dtype=np.int32)
        self.pos_left_vec = np.zeros(Nxi,dtype=np.int32)
        self.reldiff  = np.zeros(self.dimx,dtype=np.float64)
        self.reldiff_vec = np.zeros(Nxi,dtype=np.float64)
        self.weights  = np.zeros(self.ncube*2,dtype=np.float64)
        self.yi = np.zeros(self.dimy*self.Nxi,dtype=np.float64) 
        self.yi_temp = np.zeros(self.dimy,dtype=np.float64) 

        # d. position factors    
        for i in range(self.dimx-1,-1,-1):
            self.facs[i] = 1
            for j in range(i,self.dimx-1):
                self.facs[i] *= self.Nx[j+1]

        # e. add matrix
        for i in range(1,self.ncube):

            add = self.add[i*self.dimx:]
            add_prev = self.add[(i-1)*self.dimx:]

            add[:self.dimx] = add_prev[:self.dimx]
            for j in range(1,self.dimx):
                if add_prev[j] == 0:
                    add[j] = 1
                    if j > 0:
                        add[0:j] = 0
                    break
        
        add_temp = np.zeros(self.dimx,dtype=np.int32)
        for i in range(0,self.ncube):
            add = self.add[i*self.dimx:]
            for j in range(self.dimx):
                add_temp[j] = add[j]
            for j in range(self.dimx):
                add[dimx-1-j] = add_temp[j]

        # f. dimension reduction
        ncube_tot = 0
        for j in range(self.dimx):
            ncube_tot += 2**(self.dimx-1-j)
        self.cube = np.zeros(ncube_tot,dtype=np.float64)

    # evaluate
    def evaluate(self,xi):
        
        # xi is (Nxi*dimx,) numpy array
        
        for i in range(self.Nxi):

            self._evaluate(xi[i*self.dimx:],self.yi_temp)
            for l in range(self.dimy):
                self.yi[l*self.Nxi + i] = self.yi_temp[l]

        return self.yi

    def evaluate_par(self,xi):
        
        # xi is (Nxi*dimx,) numpy array
        # ONLY WORS IF values is one dimensional
        
        for i in numba.prange(self.Nxi):

            self._evaluate(xi[i*self.dimx:],self.yi[i:])

        return self.yi

    def evaluate_only_last(self,xi):
        
        # xi is (dimx,) numpy array
        # if xi_last was the last call then
        # xi[0:dimx] == xi_last[0:dimx]
        # xi[dimx] != xi_last[dimx]
         
        # a. new relative difference
        j = 0
        for d in range(1,self.dimx):
            j += self.Nx[d-1]

        xvec = self.x[j:]

        d = self.dimx-1
        self.pos_left[d] = binary_search(0,self.Nx[d],xvec,xi[d])
        
        denom =  (xvec[self.pos_left[d]+1] - xvec[self.pos_left[d]])
        self.reldiff[d] = (xi[d] - xvec[self.pos_left[d]]) / denom

        # b. initialize all to zero
        for l in range(self.dimy):
            self.yi[l] = 0

        # c. interpolate
        for i in range(self.ncube):
            index0 = self.facs[self.dimx-1]*(self.pos_left[self.dimx-1] + self.add[i*self.dimx+self.dimx-1])
            for l in range(self.dimy):
                self.yi[l] += self.y[l*self.Ny + self.indexes[i] + index0]*(self.weights[i*2+0]*(1.0-self.reldiff[self.dimx-1]))
                self.yi[l] += self.y[l*self.Ny + self.indexes[i] + index0+1]*(self.weights[i*2+1]*self.reldiff[self.dimx-1])

        return self.yi
        
    def _evaluate(self,xi,yi):
        
        # a. reldiff
        j = 0
        for d in range(self.dimx):

            if d > 0:
                j += self.Nx[d-1]
            xvec = self.x[j:]

            self.pos_left[d] = binary_search(0,self.Nx[d],xvec,xi[d])
            
            denom =  (xvec[self.pos_left[d]+1] - xvec[self.pos_left[d]])
            self.reldiff[d] = (xi[d] - xvec[self.pos_left[d]]) / denom

        # b. initialize all to zero
        for l in range(self.dimy):
            yi[l] = 0

        # c. loop through corners
        for i in range(self.ncube):

            # i. indexes
            self.indexes[i] = 0
            for j in range(0,self.dimx-1):
                self.indexes[i] += self.facs[j]*(self.pos_left[j] + self.add[i*self.dimx + j])
            
            index0 = self.facs[self.dimx-1]*(self.pos_left[self.dimx-1] + self.add[i*self.dimx+self.dimx-1])
        
            # ii. weights
            self.weights[i*2 + 0] = 1.0
            self.weights[i*2 + 1] = 1.0
            for j in range(0,self.dimx-1):

                if(self.add[i*self.dimx +j] == 1):
                    self.weights[i*2 + 0] *= self.reldiff[j]
                    self.weights[i*2 + 1] *= self.reldiff[j]
                else:
                    self.weights[i*2 + 0] *= (1.0-self.reldiff[j])
                    self.weights[i*2 + 1] *= (1.0-self.reldiff[j])

            # iii. accumulate
            for l in range(self.dimy):

                yi[l] += self.y[l*self.Ny + self.indexes[i] + index0]*(self.weights[i*2+0]*(1.0-self.reldiff[self.dimx-1]))
                yi[l] += self.y[l*self.Ny + self.indexes[i] + index0+1]*(self.weights[i*2+1]*self.reldiff[self.dimx-1])

    # evaluate montone vector
    def evaluate_monotone(self,xi,xi_vec):
        
        # xi is (dimx,) numpy array
        # xi_vec is (Nxi,) numpy array

        self._evaluate(xi,self.yi_temp)
        self.evaluate_monotone_(xi_vec)

        return self.yi

    def evaluate_monotone_(self,xi_vec):

        # a. move result to vector format
        for l in range(self.dimy):
            self.yi[l*self.Nxi + 0] = self.yi_temp[l]
        
        # b. loop through vector
        xvec = self.x[-self.Nx[self.dimx-1]:]

        self.pos_left_vec[0] = self.pos_left[self.dimx-1]
        for k in range(1,self.Nxi):

            # i. initialize at zero
            for l in range(self.dimy):
                self.yi[l*self.Nxi + k] = 0.0

            # ii. position to the left of xi in dimension 0
            i = self.pos_left_vec[k-1]
            while xi_vec[k] > xvec[i+1] and i < self.Nx[self.dimx-1]-2:
                i += 1
            self.pos_left_vec[k] = i

            # ii. relative position if xi between neighboring points
            denom = (xvec[self.pos_left_vec[k]+1] - xvec[self.pos_left_vec[k]])
            self.reldiff_vec[k] = (xi_vec[k] - xvec[self.pos_left_vec[k]]) / denom

        # c. evaluate
        for l in range(self.dimy):
            for i in range(self.ncube):
                for k in range(1,self.Nxi):

                    index0 = (self.pos_left_vec[k] + self.add[i*self.dimx + self.dimx-1])
                    self.yi[l*self.Nxi + k] += self.y[l*self.Ny + self.indexes[i]+index0]*(self.weights[i*2+0]*(1.0-self.reldiff_vec[k]))
                    self.yi[l*self.Nxi + k] += self.y[l*self.Ny + self.indexes[i]+index0+1]*(self.weights[i*2+1]*self.reldiff_vec[k])

@numba.njit(nogil=True)
def binary_search(imin,Nx,x,xi):

    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2

    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin