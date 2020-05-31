from utils import ParamsFile, HaloReader
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter
from scipy.optimize import root_scalar
from scipy.spatial import cKDTree
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


# Peak patch data
pathPrefix = "/usr/users/ellis/PeakPatch/m3p/"

# Unevolved Density fields
DensPath = "/usr/users/ellis/bin/InputFiles/512"
DensSuffix = ['/L6N3_07/axion.m.00115_L6N3_07','/L6N3_06/axion.m.00115_L6N3_06','/L6N3_01/axion.m.00115_L6N3_01']

# Cosmological parameters
z_eq = 3402
a_eq = 1/(z_eq+1)
rhoc = 2.78e-07*(1e6)**3 #Msol/cMpc^3
rho_bg = 0.267*rhoc #DM mean density Msol/cMpc^3

ma = 1e-4

def Find_z_col(z, delta_0):
    #TODO:  Modify this to take redshift of the initial conditions
    return delta_0/GrowthFactor(9e5)*(GrowthFactor(z)-1)-1.686

def thresh(z):
    return 1.686*GrowthFactor(z)/(GrowthFactor(z)-1)

def GrowthFactor(z):
    a = 1/(z+1)
    x = a/a_eq
    return 1+2/3*x

def FindDeltaSpectrum(Peaks, DensityFile, ppFile):
    # Get boxsize from pp file
    p = ParamsFile(ppFile)
    boxsize = p["boxsize"]  
    
    # Load unevolved density field   
    with h5py.File(DensityFile, 'r') as d:
        density = d['energy/redensity'][:]
        
    density = density.reshape(512,512,512)

    # Smooth on ~ 2 pixal scale - Gaussian smoothing probably not best approach
    overdensities = (density-density.mean())/density.mean()
    overdensities =  gaussian_filter(overdensities, 2)

    density = []
    all_r = []
    roots = []
    # Find delta value in *linearly* evolved density field
    
    num_peaks = len(Peaks[0,:])
    #print(num_peaks)
    delta_unEv = np.zeros(num_peaks)
    for i in range(num_peaks):
        x_cell = int(512*Peaks[0][i]/boxsize)    
        y_cell = int(512*Peaks[1][i]/boxsize) 
        z_cell = int(512*Peaks[2][i]/boxsize) 
        delta_unEv[i] = overdensities[x_cell, y_cell, z_cell]
     
    plt.hist(delta_unEv,bins=30,log=True)
    print(len(delta_unEv[delta_unEv<0])/len(delta_unEv)*100,"% less than zero")
    plt.show()

    for d in delta_unEv:
        unconverged = 0
        if d > 0:
            sol = root_scalar(Find_z_col, args = d, bracket = (1, 9e5))
            if sol.converged == True:
                roots += [sol.root]
            else:
                unconverged += 1
        else:
            print("ERROR: Delta less than zero!")
    return roots, thresh(np.asarray(roots))

def MakePeakList(ppFile, startIndex = 0, printOutput = False, massType = "normal"):
    
    assert massType in ["normal", "unstripped"], "MakePeakList(): invalid massType."
    
    # Makes a list of peaks to be used by the sub peak finder
    p = ParamsFile("inputs/" + ppFile)
    
    # Figure out the path to the directory containing the ppFile
    path = '/'.join(ppFile.split('/')[:-1])
    
    prefix =  p["output_prefix"]
    outdir = path + p["output_dir"]
    redshifts = p["redshifts"][startIndex:]
    boxsize = p["boxsize"]    

    firstFile = prefix+"final_halos_0.hdf5"
    All_Peaks = np.zeros(len(redshifts), dtype = object)

    # Make an array of all the peaks at every redshift
    for redshift_index, z in enumerate(redshifts):
        fname = outdir+"/"+prefix+"final_halos_"+repr(redshift_index+startIndex)+".hdf5"
        if printOutput == True:
            print("\tLoading file ({} of {}): {}".format(redshift_index+1, len(redshifts), fname), end = '\r')
        f = HaloReader(fname)
        
        if massType == "normal":
            All_Peaks[redshift_index] = np.vstack((f.x, f.y, f.z, f.radius, f.mass))
        elif massType == "unstripped":
            All_Peaks[redshift_index] = np.vstack((f.x, f.y, f.z, f.radius, f.unstripped_mass))
    if printOutput == True:
        print()    
    
    return All_Peaks, boxsize


def FindAllSubHalos(ppInputsFile, printOutput = False,redshift_indicies = 'all'):
    All_Peaks, boxsize = MakePeakList(ppInputsFile, printOutput = printOutput)
    num_redshifts = len(All_Peaks)
    
    # if no redshifts chosen, use all of them
    if redshift_indicies=='all':
        # Find latest redshift with peaks in it
        sizes = np.zeros(len(peak_list))
        for i in range(len(peak_list)):
            sizes[i] = peak_list[i].size
        redshift_indicies = np.arange(len(p["redshifts"]))[sizes>0]
    
    trees = [cKDTree(All_Peaks[i][0:3].T, boxsize = boxsize) for i in redshift_indicies]
    
    final_peaks = All_Peaks[-1]

    for redshift_index in redshift_indicies[:-1]:
        
        # Find nearest neighbor for every peak at this redshift
        query = trees[redshift_index+1].query(All_Peaks[redshift_index][0:3, :].T, k=1, eps=0)
        dists = query[0]
        # take radii of every peak at this redshift
        radii = All_Peaks[redshift_index][3,:]
        
        # If at the previous redshift, there are no peaks within the radius of the peak at this redshift,
        # then this is the first instance of that peak.
        NewPeaks = All_Peaks[redshift_index].T[dists>radii].T
        
        if printOutput == True:
            print("zi = {}: {} new peaks.".format(redshift_index,len(NewPeaks[0,:])))
            
        # Add theses peaks to the store
        final_peaks = np.concatenate((final_peaks,NewPeaks),axis=1)
    if printOutput == True:
        print(final_peaks.shape, All_Peaks[-1].shape)
      
    return final_peaks

def BuildMergerTree(peak_list, pp_file, redshift_indicies='all', final_halos_indicies = 'all', printOutput = False):
    '''
    Builds lists of progenitor peaks for one (or multiple) final peak(s).
       Progenitors are defined to be all peaks contained within the comoving radius of the product (final) halo.
    '''
    # TODO: Add function description
    # TODO: Enable multi-theading
    
    p = ParamsFile("inputs/" + pp_file)
    boxsize = p["boxsize"]  
    
    # if no redshifts chosen, use all of them
    if redshift_indicies=='all':
        # Find latest redshift with peaks in it
        sizes = np.zeros(len(peak_list))
        for i in range(len(peak_list)):
            sizes[i] = peak_list[i].size
        redshift_indicies = np.arange(len(peak_list))[sizes>0]
        
    if printOutput == True:
        print("\tFinal redshift index {} out of {}".format(max(redshift_indicies), len(peak_list)))
        print("\ti.e. Earlist halo at z = {}".format(p["redshifts"][max(redshift_indicies)]))
        
    trees = [cKDTree(peak_list[i][0:3].T, boxsize = boxsize) for i in redshift_indicies]
    
    if final_halos_indicies == 'all':
        final_halos_indicies = np.arange(len(peak_list[0].T))
    # if only single final halo chosen, turn it into an array so that the code works
    
    elif type(final_halos_indicies) == int:
        final_halos_indicies = np.array([final_halos_indicies])
        
    merger_trees = np.zeros(len(final_halos_indicies), dtype = object)
    # Loop over all selected final halos
    for hi, final_halo_index in enumerate(final_halos_indicies):
        # Find all other sub-peaks 
        final_radius = peak_list[0][3,final_halo_index] 
        peaks = np.zeros(len(redshift_indicies), dtype=object)  
        peaks[0] = np.vstack(np.asarray(peak_list[0][:, final_halo_index])).T
        for ri, redshift_index in enumerate(redshift_indicies[:-1]):
            new_peaks = []
            if peaks[redshift_index].size>0:
                peak_count = 10
                query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=peak_count, eps=0)
                dists = query[0]
                radius = final_radius #peaks[redshift_index][parent_index, 3]

                while len(dists[dists<radius]) == peak_count:
                    peak_count = peak_count+100
                    query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=peak_count, eps=0)
                    dists = query[0]
                    #query = trees[redshift_index+1].query(peaks[redshift_index][parent_index, 0:3], k=50, eps=0)
                #print(radius, ":", dists[0:3])


                for i, index in enumerate(query[1][dists<radius]):
                    #print(peak_list[redshift_index+1][:, index])
                    new_peaks.append(peak_list[redshift_index+1][:, index])  

                # Check all the new peaks are within final halo
                if len(new_peaks)>0:
                    query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=len(new_peaks), eps=0)
                    dists = np.array(query[0])
                    insideFinal = len(dists[dists<final_radius])
                    assert insideFinal == len(new_peaks), '''
{} peaks found but only {} are within final halo radius
final_radius = {}
dists = {}
'''.format(len(new_peaks),insideFinal,final_radius, dists)
                        
            new_peaks = np.asarray(new_peaks)
            if len(new_peaks) == 1:
                peaks[redshift_index+1] = np.vstack(np.asarray(new_peaks))
            elif new_peaks.size>0:
                peaks[redshift_index+1] = np.vstack(np.asarray(new_peaks))
            else:
                peaks[redshift_index+1] = np.asarray([])
            if printOutput == True:
                # print progress
                print("\tHalo {} of {}: {} complete of {}".format(hi,len(final_halos_indicies), 
                                                                  ri+1, len(redshift_indicies[:-1])), end = '\r')
        merger_trees[hi] = peaks
    if printOutput == True:
        # print new line
        print()
        # Store merger tree
        
    return merger_trees

def volInt(r1, r2, d):
    """
    Calculates volume of intersecting region of two spheres or radius r1 and r2
    seperated by a distance d
    
    https://en.wikipedia.org/wiki/Spherical_cap
    """
    
    # Ensure values are positive
    assert (r1 >= 0)*(r2 >= 0)*(d >= 0), "Input values must be positive!"
    
    if d > r1+r2:
        Vol = 0 
    elif d > abs(r1 - r2):
        A = np.pi/(12*d)
        B = (r1+r2-d)**2
        C = (d**2+2*d*(r1+r2)-3*(r1-r2)**2)
        
        Vol = A*B*C
 
    elif d <= abs(r1 - r2):
        Vol = 4/3*np.pi*min(r1, r2)**3
    
    return Vol
    
def midPoint(coord1, coord2, radius1, radius2):
    if coord1 > coord2:
        midpoint = 0.5*(coord1+coord2+radius2-radius1)
    if coord1 < coord2:
        midpoint = 0.5*(coord1+coord2+radius1-radius2)
    elif coord1 == coord1:
        midpoint = coord1
        
    return midpoint
    
def intMid(peak1, peak2):
    '''
    Calculate the coordinates of the center of the intersection
    between two spheres with coordinates (x1, y1, z1) and (x2, y2, z2)
    and radii r1 and r2 respectively.
    '''
    x1, y1, z1, r1, M1 = peak1
    x2, y2, z2, r2, M2 = peak2
    
    xc = midPoint(x1, x2, r1, r2)
    yc = midPoint(y1, y2, r1, r2)
    zc = midPoint(z1, z2, r1, r2)
    
    return xc, yc, zc

def BuildMergerTree2(peak_list, pp_file, redshift_indicies='all', final_halos_indicies = 'all', 
                     effectiveCoords = False, printOutput = False):
    '''
    Builds lists of progenitor peaks for one (or multiple) final peak(s).
       Progenitors are defined to be all peaks contained within the comoving radius of the product (final) halo.
       
       Trying to build new version that counts all mass within final radius
    '''
    # TODO: Add function description
    # TODO: Enable multi-theading
    
    p = ParamsFile("inputs/" + pp_file)
    boxsize = p["boxsize"]  
    
    # if no redshifts chosen, use all of them
    if redshift_indicies=='all':
        # Find latest redshift with peaks in it
        sizes = np.zeros(len(peak_list))
        for i in range(len(peak_list)):
            sizes[i] = peak_list[i].size
        redshift_indicies = np.arange(len(peak_list))[sizes>0]
        
    if printOutput == True:
        print("\tFinal redshift index {} out of {}".format(max(redshift_indicies), len(peak_list)))
        print("\ti.e. Earlist halo at z = {}".format(p["redshifts"][max(redshift_indicies)]))
        
    trees = [cKDTree(peak_list[i][0:3].T, boxsize = boxsize) for i in redshift_indicies]
    
    if final_halos_indicies == 'all':
        final_halos_indicies = np.arange(len(peak_list[0].T))
    # if only single final halo chosen, turn it into an array so that the code works
    elif type(final_halos_indicies) == int:
        final_halos_indicies = np.array([final_halos_indicies])
        
    merger_trees = np.zeros(len(final_halos_indicies), dtype = object)
    
    # Loop over all selected final halos
    for hi, final_halo_index in enumerate(final_halos_indicies):
        # Find all other sub-peaks 
        final_radius = peak_list[0][3,final_halo_index] 
        
        # max radius at this redshift
        max_radius = max(peak_list[0][3, :])
        
        peaks = np.zeros(len(redshift_indicies), dtype=object)  
        peaks[0] = np.vstack(np.asarray(peak_list[0][:, final_halo_index])).T
        
        for ri, redshift_index in enumerate(redshift_indicies[:-1]):
            new_peaks = []
            
            if peaks[redshift_index].size>0:
                peak_count = 10
                query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=peak_count, eps=0)
                dists = query[0]
    
                # Keep finding more peaks until some are further away than final_radius + max_radius
                while len(dists[dists < final_radius + max_radius]) == peak_count:
                    peak_count = peak_count+100
                    query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=peak_count, eps=0)
                    dists = query[0]

                
                for i, index in enumerate(query[1][dists < final_radius + max_radius]):
                    # Check if any overlap
                    subPeakRadius = peak_list[redshift_index+1][3, index]
                    
                    if dists[i]<final_radius+subPeakRadius:
                        
                        # Calculate volume of overlap and the mass of the sub-halo in that region 
                        volumeOverlap = volInt(final_radius, subPeakRadius, dists[i])
                        assert volumeOverlap >= 0, "Volume must be positive"
                        
                        # Calculate effective mass and density
                        subPeakMass = peak_list[redshift_index+1][4, index]
                        subPeakVolume = 4/3*np.pi*subPeakRadius**3
                        massInside = volumeOverlap/subPeakVolume*subPeakMass 
                        r_effective = (3*volumeOverlap/(4*np.pi))**(1/3)
                       
                        # Store sub-halo with this new mass and effective radius
                        subPeak = peak_list[redshift_index+1][:, index]
                        subPeak[4] = massInside
                        subPeak[3] = r_effective
                        
                        if effectiveCoords == True:
                            # Calculate center of overlap
                            xc, yc, zc = intMid(peak_list[0][:,final_halo_index],
                                               peak_list[redshift_index+1][:, index])
                            subPeak[0:3] = xc, yc, zc
                        
                        new_peaks.append(subPeak)  

                # Check all the new peaks are within 2* final halo radius
                if len(new_peaks)>0:
                    query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=len(new_peaks), eps=0)
                    dists = np.array(query[0])
                    insideFinal = len(dists[dists<2*final_radius])
                    assert insideFinal == len(new_peaks), '''
{} peaks found but only {} are within 2* final halo radius
final_radius = {}
dists = {}
'''.format(len(new_peaks),insideFinal,final_radius, dists)
                        
            new_peaks = np.asarray(new_peaks)
            if len(new_peaks) == 1:
                peaks[redshift_index+1] = np.vstack(np.asarray(new_peaks))
            elif new_peaks.size>0:
                peaks[redshift_index+1] = np.vstack(np.asarray(new_peaks))
            else:
                peaks[redshift_index+1] = np.asarray([])
            if printOutput == True:
                # print progress
                print("\tHalo {} of {}: {} complete of {}".format(hi,len(final_halos_indicies), 
                                                                  ri+1, len(redshift_indicies[:-1])), end = '\r')
        # Store merger tree
        merger_trees[hi] = peaks
    if printOutput == True:
        # print new line
        print()
        
    return merger_trees

def MoveOutOfBounds(merger_list, boxsize, printOutput=False):
    if printOutput == True:
        print("Moving out of bounds...")

    for j, peaks in enumerate(merger_list[1:]):
        if printOutput == True:
            print("redshift index: {}".format(j+1))
        if peaks.size>0:
            diffs = peaks[:, 0:3]-merger_list[0][0][0:3]
            merger_list[j+1][:,0:3][diffs>boxsize/2] -= boxsize
            merger_list[j+1][:,0:3][diffs<-boxsize/2] += boxsize
            if printOutput == True:
                print("{} coords shifted".format(np.sum(((diffs>boxsize/2)+(diffs<-boxsize/2)).flatten())))
    return merger_list

def pruneLowMasses(merger_list, min_mass, printOutput=False):
    '''Removes low mass halos from list '''
    if printOutput == True:
        print("Removing small halos...")

    for j, peaks in enumerate(merger_list[1:]):
        if printOutput == True:
            print("redshift index: {}".format(j+1))
        if peaks.size>0:
            masses = peaks[:, 4]
            merger_list[j+1] = merger_list[j+1][masses > min_mass]
                
    return merger_list

    

def plotMergerTree(merger_list, pp_file,startIndex=0, printOutput = False, 
                   cmap = 'gnuplot_r', font_size = 15, log = False, colorbar = False, 
                   colorbar_title = None, min_mass = 0):
    # TODO: Add function description
    # TODO: Sort heights of peaks based on y-axis posn or something.
    # TODO: Add minimum mass filter
            
    # Filter out low mass halos
    merger_list = pruneLowMasses(merger_list, min_mass, printOutput)
    
    # only plot for redshifts with peaks in them
    last_index = 0
    for i in range(len(merger_list)-1):
        if merger_list[i].size > 0:
            last_index += 1
    
    p = ParamsFile("inputs/" + pp_file)
    redshifts = p["redshifts"][startIndex:]
    boxsize = p["boxsize"]
   
    # Move peaks to account for periodic boundary conditions
    merger_list = MoveOutOfBounds(merger_list, boxsize, printOutput)
    
    # plotting info
    matplotlib.rc('font', size=font_size)
    #print(merger_list[-1][:,4]
    # Colour on mass
    colormap = cm = plt.get_cmap(cmap) 
    cNorm  = colors.Normalize(np.log10(max(1e-15, min_mass)), np.log10(merger_list[0][0,4]))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
    
    fig = plt.figure(figsize = (7, 5))
    for i in range(last_index):
        
        if printOutput == True:
            print("redshift index: {}".format(i))
            
        # Check each peak for peaks at the earlier redshift
        for j in range(merger_list[i].shape[0]):
            if i<len(merger_list)-1 and merger_list[i+1].size>0:
                #print(merger_list[i+1][:,0:3],"\n", merger_list[i+1][:,3])
                dists = np.sqrt(np.sum((merger_list[i][j,0:3]-merger_list[i+1][:,0:3])**2,axis=1))
                inside_mask = dists<merger_list[i][j,3]#+merger_list[i+1][:,3]/3
                for index in np.where(inside_mask)[0]:
                    plt.plot([redshifts[i],redshifts[i+1]], [j-merger_list[i].shape[0]/2,index-merger_list[i+1].shape[0]/2], 'k--')
                    #print(index, )
                    
            ms = 30*merger_list[i][j,3]/merger_list[0][0,3]
            colorVal = scalarMap.to_rgba(np.log10(merger_list[i][j,4]))    
            plt.plot(redshifts[i], j-merger_list[i].shape[0]/2,'o',
                     ms = ms, color = colorVal)    
            # 
    if log == True:
        plt.xscale('log')
    plt.yticks([])     
    plt.xlim(redshifts[last_index], redshifts[0]*0.8)
    plt.xlabel("Redshift, $z$")
    
    if colorbar == True:
        scalarMap.set_array([])
        cbar = fig.colorbar(scalarMap)
        if colorbar_title != None:
            assert type(colorbar_title) == str, "Error: colorbar_title must be a string"
            cbar.ax.set_ylabel(colorbar_title)
            
    return fig


def plotMergerPatches(merger_list, pp_file, printOutput = False, cmap = 'gnuplot'):
    
    last_index = 0
    for i in range(len(merger_list)-1):
        if merger_list[i].size > 0:
            last_index += 1
   
    p = ParamsFile("inputs/" + pp_file)
    redshifts = p["redshifts"]  
    boxsize = p["boxsize"]
    matplotlib.rc('font', size=15)
    # Move peaks to account for periodic boundary conditions
    merger_list = MoveOutOfBounds(merger_list, boxsize, printOutput)
    
    import matplotlib.colors as colors
    fig = plt.figure(figsize = (11,5))
    colormap = cm = plt.get_cmap(cmap) 
    cNorm  = colors.Normalize(redshifts[0], redshifts[last_index])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
    
    ax1 = fig.add_subplot(121)

    ax1.set_xlim(merger_list[0][0,0]-merger_list[0][0,3], merger_list[0][0,0]+merger_list[0][0,3])
    ax1.set_ylim(merger_list[0][0,1]-merger_list[0][0,3], merger_list[0][0,1]+merger_list[0][0,3])

    #ax1.set_xlim(-boxsize, boxsize)
    #ax1.set_ylim(0, 2*boxsize)

    ax2 = fig.add_subplot(122)
    ax2.set_xlim(merger_list[0][0,0]-merger_list[0][0,3], merger_list[0][0,0]+merger_list[0][0,3])
    ax2.set_ylim(merger_list[0][0,2]-merger_list[0][0,3], merger_list[0][0,2]+merger_list[0][0,3])

    colors = ['b','r','g','m','y','k']
    for i in range(len(merger_list)):
        colorVal = scalarMap.to_rgba(redshifts[i])
        peaks = merger_list[i]
        #print(peaks, peaks.size)
        for peak in peaks:
            pxy = mpatches.Circle((float(peak[0]), float(peak[1])), peak[3], alpha = 0.2, color = colorVal)
            ax1.add_patch(pxy)

            pxz = mpatches.Circle((float(peak[0]), float(peak[2])), peak[3], alpha = 0.2, color = colorVal)
            ax2.add_patch(pxz)
        else:
            # No peak here
            pass
        
def FindCollapseRedshift(merger_tree, thresh_frac, pp_file, 
                         startIndex = 0, printOutput = False, interp = "None"):
    last_index = 0
#     for i in range(len(merger_tree)):
#         if merger_tree[i].size > 0:
#             last_index += 1
    
    p = ParamsFile("inputs/" + pp_file)
    redshifts = p["redshifts"][-len(merger_tree):]
    #print(redshifts[0], redshifts[-1])
    FinalMass = merger_tree[0][0, 4]
    
    ProgMass = np.zeros(len(redshifts))
    for ri in range(len(redshifts)):
        if merger_tree[ri].size > 0:
            ProgMass[ri] = np.sum(merger_tree[ri][:, 4][merger_tree[ri][:, 4]>=thresh_frac*FinalMass])
    
    if interp == "None":
        # No interpolation:
        # just find maximum redshift which meets the M>M_tot/2 requirement
        if printOutput == True:
            print(max(ProgMass), FinalMass/2)
        CollapseRedshift = max(redshifts[ProgMass>=FinalMass/2])
    elif interp == "Linear":
        print("Error: Linear iterpolation not implemented yet")
        return 0
                     
    return CollapseRedshift, ProgMass, redshifts