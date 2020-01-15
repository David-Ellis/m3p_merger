 
import numpy as np
class HaloReader(object):
    def __init__(self,fname):
        import h5py
        from astropy import units as u
        with h5py.File(fname) as f:
            self.x = f["x"][:]
            self.y = f["y"][:]
            self.z = f["z"][:]
            self.mass = f["mass"][:]
            self.radius = f["radius"][:]
            self.detected_at = f["detected_at"][:]
            #read header information
            self.redshift = f["Parameters"].attrs["redshift"]
            self.boxsize = f["Parameters"].attrs["boxsize"]*u.Mpc
            self.little_h = f["Parameters"].attrs["little_h"]
    def mask_mass(self,Min=1e-10,Max=1e20):
        return (self.mass>=Min) & (self.mass<Max)
    def size(self):
        return self.x.size


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
