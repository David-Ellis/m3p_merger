import numpy as np
from astropy import units as u
import h5py

class HaloReader(object):
    def __init__(self,fname=None,from_parent=None,mask=None):
        
        from astropy import units as u
        if fname is None and from_parent is None:
            raise NameError("You need to provide either a file name or a parent object")
        if fname is not None:
            with h5py.File(fname) as f:
                self.x = f["x"][:]
                self.y = f["y"][:]
                self.z = f["z"][:]

                self.dispx = f["dispx"][:]
                self.dispy = f["dispy"][:]
                self.dispz = f["dispz"][:]

                self.mass = f["mass"][:]
                self.radius = f["radius"][:]
                self.detected_at = f["detected_at"][:]
                #read header information
                self.redshift = f["Parameters"].attrs["redshift"]
                self.boxsize = f["Parameters"].attrs["boxsize"]*u.Mpc
                self.little_h = f["Parameters"].attrs["little_h"]
                self.n_cell = f["Parameters"].attrs["n_cell"]
                self.D = f["Parameters"].attrs["D"]
                self.f = f["Parameters"].attrs["f"]
                self.Hz = f["Parameters"].attrs["Hz"]
        else:
            if mask is None:
                mask = np.ones_like(from_parent.x,dtype=bool)
            self.x = from_parent.x[mask]
            self.y = from_parent.y[mask]
            self.z = from_parent.z[mask]

            self.dispx = from_parent.dispx[mask]
            self.dispy = from_parent.dispy[mask]
            self.dispz = from_parent.dispz[mask]

            self.mass = from_parent.mass[mask]
            self.radius = from_parent.radius[mask]
            self.detected_at = from_parent.detected_at[mask]
            #read header information
            self.redshift = from_parent.redshift
            self.boxsize = from_parent.boxsize
            self.little_h = from_parent.little_h
            self.n_cell = from_parent.n_cell
            self.D = from_parent.D
            self.f = from_parent.f
            self.Hz = from_parent.Hz
            
            self.dx = self.boxsize/self.n_cell
    def write(self,fname):
        with h5py.File(fname,"w") as f:
            f.create_dataset("x",data=self.x)
            f.create_dataset("y",data=self.y)
            f.create_dataset("z",data=self.z)

            f.create_dataset("dispx",data=self.dispx)
            f.create_dataset("dispy",data=self.dispy)
            f.create_dataset("dispz",data=self.dispz)

            f.create_dataset("mass",data=self.mass)
            f.create_dataset("radius",data=self.radius)
            f.create_dataset("detected_at",data=self.detected_at)
            
            
            f.create_dataset("max_redshift",data=np.zeros_like(self.mass))
            f.create_dataset("delta",data=np.zeros_like(self.mass))
            #write header information
            f.create_group("Parameters")
            f["Parameters"].attrs["redshift"] = self.redshift
            f["Parameters"].attrs["boxsize"] = self.boxsize.value
            f["Parameters"].attrs["little_h"] = self.little_h 
            f["Parameters"].attrs["n_cell"] = self.n_cell
            f["Parameters"].attrs["D"] = self.D
            f["Parameters"].attrs["f"] = self.f
            f["Parameters"].attrs["Hz"] = self.Hz
        
    def __getattribute__(self,name):
        if(name=="position"):
            pos = np.array([self.x+self.D*self.dispx, self.y+self.D*self.dispy, self.z+self.D*self.dispz]).T*u.Mpc
            pos[pos<0.]+=self.boxsize
            pos[pos>=self.boxsize]-=self.boxsize
            return pos
        elif(name=="velocity"):
            a = 1./(1.+self.redshift)
            return self.f * self.Hz * a / self.little_h * np.array([self.dispx, self.dispy, self.dispz]).T * u.km/u.s
        elif(name=="index"):
            return self.get_index()
        else:
            return super(HaloReader,self).__getattribute__(name)
    def get_index(self):
        dx = self.boxsize.value/self.n_cell
        idx = np.zeros_like(self.x,dtype=np.int64)
        for i, coord in enumerate([self.x, self.y, self.z]):
            temp = np.round((coord-dx/2.)/dx).astype(np.int64)
            idx += self.n_cell**i*temp
        return idx
            
    def mask_mass(self,Min=1e-10,Max=1e20):
        return (self.mass>=Min) & (self.mass<Max)
    def size(self):
        return self.x.size
    def plot_dndm(self,ax=None,Mmin=11.,Mmax=16.,nbins=50,label="m3p"):
        import matplotlib.pyplot as plt
        if(ax is None):
            ax = plt.gca()
        bins = np.logspace(Mmin,Mmax,nbins)
        H,xedges = np.histogram(self.mass,bins=bins)
        H= H/np.diff(xedges)/(self.boxsize)**3
        ax.plot(xedges[:-1],H,label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
    def plot_2pcf(self,Mmin=0.,Mmax=1e20,ax=None,rmin=None,rmax=None,niterations=1,nbins=50,label="m3p",binning="Log"):
        import matplotlib.pyplot as plt
        if(ax is None):
            ax = plt.gca()
        if(rmin is None):
            rmin = self.dx.value*4.
        if(rmax is None):
            rmax = self.boxsize.value/4.
        mask = self.mask_mass(Mmin,Mmax)
        r, xi, _ = self.get_2pcf(rmin,rmax,nbins,niterations=niterations,mask=mask,binning=binning)
        ax.plot(r,r**2*xi,label=label)

    def print_halo(self,i):
        print(self.x[i],self.y[i],self.z[i],self.mass[i],self.radius[i],self.detected_at[i])
    def sort_by(self,keys):
        nd = np.lexsort(keys)
        self.x = self.x[nd]
        self.y = self.y[nd]
        self.z = self.z[nd]
        self.dispx = self.dispx[nd]
        self.dispy = self.dispy[nd]
        self.dispz = self.dispz[nd]
        self.mass = self.mass[nd]
        self.radius = self.radius[nd]
        self.detected_at = self.detected_at[nd]
    def get_2pcf(self,rmin, rmax, nbins, estimator="default", mask=None, niterations=1,return_components=False,binning="Log"):
        import treecorr
        
        if(mask is None):
            mask=np.ones_like(self.x,dtype=bool)
        pos = self.position[mask].value
        
        x1=pos[:,0]
        y1=pos[:,1]
        z1=pos[:,2]
            
        cat1 = treecorr.Catalog(x=x1, y=y1, z=z1)
        dd = self.get_dd(cat1, rmin, rmax, nbins,binning=binning)
        rr, dr = self.get_rr_dr(cat1,rmin,rmax,nbins,niterations=niterations,binning=binning)
        xi, varxi = dd.calculateXi(rr,dr)
        r = (dd.right_edges + dd.left_edges)/2.

        if(return_components):
            return r, xi, varxi, dd, rr, dr
        else:
            return r, xi, varxi

    def get_dd(self,cat,rmin,rmax,nbins,binning="Log"):
        import treecorr

        dd = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins, bin_slop=0,
                                    xperiod=self.boxsize.value, yperiod=self.boxsize.value,
                                    zperiod=self.boxsize.value,bin_type=binning,metric="Periodic")
        dd.process(cat, metric='Periodic')
        return dd
    def get_rr_dr(self,cat,rmin,rmax,nbins,niterations=1,binning="Log"):
        import treecorr
        from numpy.random import random_sample, choice
        rrcats = []
        for i in range(0,niterations):
            nhalos = cat.ntot
            boxsize = self.boxsize.value
            x2 = (random_sample(nhalos*2)) * boxsize
            y2 = (random_sample(nhalos*2)) * boxsize
            z2 = (random_sample(nhalos*2)) * boxsize
            cat2 = treecorr.Catalog(x=x2, y=y2, z=z2)
            rrcats.append(cat2)
        
        dr = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins, bin_slop=0,
                            xperiod=boxsize, yperiod=boxsize,zperiod=boxsize,bin_type=binning,metric="Periodic")
        rr = treecorr.NNCorrelation(min_sep=rmin, max_sep=rmax, nbins=nbins, bin_slop=0,
                            xperiod=boxsize, yperiod=boxsize,zperiod=boxsize,bin_type=binning,metric="Periodic")
        dr.process(rrcats, cat, metric='Periodic')
        rr.process(rrcats, metric='Periodic')
        
        return rr,dr
    
        
        
from collections import OrderedDict

class ParamsFile(OrderedDict):
    '''copied from params_file in Iltis, https://github.com/cbehren/Iltis/blob/master/python/params_file.py'''
    def __init__(self,fname=None):
        self.float_types = ["boxsize","max_filter_radius","peak_threshold"]
        self.bool_types = []
        self.int_types = ["ics.seed","n_cell","max_grid_size","nfilters"]
        self.float_vector_types = ["redshifts","filter_scales"]
        self.int_vector_types = []
        
        if(fname is not None and isinstance(fname,str)):
            super(ParamsFile,self).__init__()
            self.read(fname)
        else:
            super(ParamsFile,self).__init__(fname)
            
            
    def update(self):
        self.write(self.original_name)
    def read(self,fname):
        self.original_name = fname
        self.clear()
        los_count = 0
        with open(fname,"r") as f:
            lines = f.readlines()
            for line in lines:
                line=line.rstrip()
                line=line.lstrip()
                if(len(line)<1 or line[0] is '#'):
                    continue
                line=line.split('=')
                if(len(line)>1):
                    key=line[0].strip()
                    value = line[1].split("#")[0]
                    value = self.type_conversion(key,value.strip())
                    if("line_of_sight" in key):
                        key += "%"+repr(los_count)
                        los_count+=1
                    self[key]=value
    def write(self,fname,comment=None):
        with open(fname,"w") as f:
            f.write("#file written by params_file class\n")
            if(comment is not None):
                f.write("#"+comment+"\n")
            for key,value in  self.iteritems():
                if("%" in key):
                    key = key.split("%")[0]
                if(isinstance(value,str)):
                    v = value
                elif(isinstance(value,np.ndarray)):
                    v = ""
                    for d in value:
                        v+=repr(d)+" "
                else:
                    v = repr(value)
                if(isinstance(value,bool)):
                    v = v.lower()
                f.write(key+"="+v+"\n")
    def type_conversion(self,key,value):
        if key in self.float_types:
            return float(value)
        if key in self.int_types:
            try:
                v = int(value)
            except ValueError:
                print("Warning:",key,"should be integer, but is",value)
                v= int(round(float(value)))
            
            return v
        if key in self.float_vector_types:
            return np.array([float(v) for v in value.split()])
        if key in self.int_vector_types:
            return np.array([int(v) for v in value.split()])
        if key in self.bool_types:
            if(value in ["true","True"]):
                   ret = True
            elif(value in ["false","False"]):
                   ret = False
            else:
                raise NameError("Could not convert parameter to boolean!")
            return ret
        
        return value

import os
   
class m3pSimulation(object):
    def __init__(self,fname):
        self.params = ParamsFile(fname)
        if(not "redshifts" in self.params):
            self.params["redshifts"]=[0.]
        self.redshifts = self.params["redshifts"]
        
        self.output_prefix = ""
        if("output_prefix" in self.params):
            self.output_prefix = self.params["output_prefix"]
        self.output_dir = "out"
        if("output_dir" in self.params):
            self.output_dir = self.params["output_dir"]
        self.basedir = os.path.dirname(os.path.abspath(fname))
        self.basename = os.path.join(self.basedir,self.output_dir,self.output_prefix)
    def get_halos(self,redshift):
        for i in range(0,len(self.redshifts)):
            if(self.redshifts[i]==redshift):
                    return HaloReader(self.basename+"final_halos_"+repr(i)+".hdf5")
        raise NameError("Could not find halos for redshift "+repr(redshift))
    def get_powerspectrum(self):
        k, ps, _, _ = np.loadtxt(self.basename+"PS.dat",unpack=True)
        return k,ps
                
                
### some helper routines

def diff_halos(fname1,fname2,return_all=False,return_indizes=False):
    from astropy import units as u
    #compares masses of halos at same position between two files. useful for debugging.
    #returns the difference in mass between halos located at the same position. 
    #only makes sense if both runs had the same underlying density field.
    h1 = HaloReader(fname1)
    h2 = HaloReader(fname2)
    if(h1.size()<h2.size()):
        h1, h2 = h2, h1
    nh1 = h1.size()
    nh2 = h2.size()
    print(nh1, nh2)
    h1.sort_by([h1.index])
    h2.sort_by([h2.index])
    difference = (np.zeros_like(h1.x)-np.sqrt(-1))
    index2 = h2.index
    index1 = h1.index
    j=0
    diff_index1 = []
    diff_index2 = []
    for i in range(0,h1.size()):
        if(j >= nh2):
            break
        idx1 = index1[i]        
        idx2 = index2[j]
        if(idx2==idx1):
            difference[i] = (h1.mass[i]-h2.mass[j])/h1.mass[i]
            diff_index1.append(i)
            diff_index2.append(j)
            j=j+1
        else:
            found = False
            j=j+1
            idx2 = index2[j]
            print(i,j,idx1,idx2)
            while(idx2<idx1 and j<nh2):    
                idx2 = index2[j]
                if(idx2==idx1):
                    difference[i] = (h1.mass[i]-h2.mass[j])/h1.mass[i]
                    diff_index1.append(i)
                    diff_index2.append(j)
                    found = True
                    break
                j=j+1
            else:
                j=j-1
    if(not return_all):
        difference = difference[difference==difference]
    if(return_indizes):
        return difference, diff_index1, diff_index2
    else:
        return difference
                
                
        
