import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
import matplotlib.patches as mpatches
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from m3p_merger.utils import ParamsFile, HaloReader

def MakePeakList(ppFile, path_prefix, startIndex = 0, printOutput = False, massType = "normal"):
    
    assert massType in ["normal", "unstripped"], "MakePeakList(): invalid massType."
    
    # Makes a list of peaks to be used by the sub peak finder
    ppFile_path = path_prefix + "/inputs/" + ppFile
    
    #print(ppFile_path)
    p = ParamsFile(ppFile_path)
    
    # Figure out the path to the directory containing the ppFile
    #path = '/'.join(ppFile.split('/')[:-1])
    
    prefix =  p["output_prefix"]
    outdir = path_prefix +"/"+ p["output_dir"]
    redshifts = p["redshifts"][startIndex:]
    boxsize = p["boxsize"]    

    #firstFile = path_prefix + prefix+"final_halos_0.hdf5"
    All_Peaks = np.zeros(len(redshifts), dtype = object)

    # Make an array of all the peaks at every redshift
    for redshift_index, z in enumerate(redshifts):
        fname = outdir+"/"+prefix+"final_halos_"+repr(redshift_index+startIndex)+".hdf5"
        #print(fname)
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
    
    # if no redshifts chosen, use all of them
    if redshift_indicies=='all':
        p = ParamsFile(ppInputsFile)
        # Find latest redshift with peaks in it
        sizes = np.zeros(len(All_Peaks))
        for i in range(len(All_Peaks)):
            sizes[i] = All_Peaks[i].size
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

def BuildMergerTree_OLD(peak_list, pp_file, path_prefix, redshift_indicies='all', final_halos_indicies = 'all', 
                    printOutput = False):
    '''
    Builds lists of progenitor peaks for one (or multiple) final peak(s).
       Progenitors are defined to be all peaks contained within the comoving radius of the product (final) halo.
    
    !! Use BuildMergerTree as the results are more stable. !!
    '''
    # TODO: Add function description
    # TODO: Enable multi-theading
    ppFile_path = path_prefix +"/inputs/" + pp_file
    
    #print(ppFile_path)
    p = ParamsFile(ppFile_path)
    
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
    print("Done.")
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

def BuildMergerTree(peak_list, ppFile, path_prefix, redshift_indicies='all', final_halos_indicies = 'all', 
                     effectiveCoords = False, printOutput = False):
    '''
    Builds lists of progenitor peaks for one (or multiple) final peak(s).
       Progenitors are defined to be all peaks contained within the comoving radius of the product (final) halo.
       
       Trying to build new version that counts all mass within final radius
    '''
    # TODO: Add function description
    # TODO: Enable multi-theading
    
    ppFile_path = path_prefix + "/inputs/" + ppFile

    p = ParamsFile(ppFile_path)
    boxsize = p["boxsize"]  
    
    # if no redshifts chosen, use all of them
    if redshift_indicies=='all':
        # Find latest redshift with peaks in it
        sizes = np.zeros(len(peak_list))
        for i in range(len(peak_list)):
            sizes[i] = peak_list[i].size
        redshift_indicies = np.arange(len(peak_list))[sizes>0]
        
    if printOutput == True:
        print("\tFinal redshift index {} out of {}".format(max(redshift_indicies), len(peak_list)-1))
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
            #print("Redshift index: {} ".format(ri), end = "   ")
            #print("# peaks at this z: {}".format(len(peak_list[redshift_index+1][0,:])))
            
            new_peaks = []
            
            if peaks[redshift_index].size>0:
                peak_count = 10
                query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=peak_count, eps=0)
                
                # maxIndex prevents double counting and taking more peaks than what exist
                maxIndex = min(len(query[0]), len(peak_list[redshift_index+1][0,:]))
                #print("maxIndex =", maxIndex)
                dists = query[0][:maxIndex]
                
                subPeakRadii = peak_list[redshift_index+1][3, :][query[1][:maxIndex]].copy()
                
                # Keep finding more peaks until some are further away than final_radius + max_radius
                while len(dists[dists < final_radius + max_radius]) == peak_count:
                    peak_count = peak_count+100
                    query = trees[redshift_index+1].query(peak_list[0][0:3,final_halo_index], k=peak_count, eps=0)
                    
                    maxIndex = min(len(query[0]), len(peak_list[redshift_index+1][0,:]))
                    #print("maxIndex =", maxIndex)
                    dists = query[0][:][:maxIndex]
                    #print("len(peak_list):", len(peak_list[redshift_index+1][0, :][query[1]]), "len(query[1]):", len(query[1]))
                    subPeakRadii = peak_list[redshift_index+1][3, :][query[1][:maxIndex]].copy()
                
                for i, index in enumerate(query[1][:maxIndex][dists < final_radius + subPeakRadii]):
                    # Check if any overlap
                    subPeakRadius = peak_list[redshift_index+1][3, index]
                   
                    if dists[i] < final_radius+subPeakRadius:
                        
                        # Calculate volume of overlap and the mass of the sub-halo in that region 
                        volumeOverlap = volInt(final_radius, subPeakRadius, dists[i])
                        assert volumeOverlap >= 0, "Volume must be positive"
                        
                        # Calculate effective mass and density
                        subPeakMass = peak_list[redshift_index+1][4, index]
                        subPeakVolume = 4/3*np.pi*subPeakRadius**3
                        massInside = volumeOverlap/subPeakVolume*subPeakMass 
                        r_effective = (3*volumeOverlap/(4*np.pi))**(1/3)
                        #print("r_eff: {:.3}".format(r_effective))
                        # Store sub-halo with this new mass and effective radius
                        subPeak = peak_list[redshift_index+1][:, index].copy()
                        subPeak[4] = massInside
                        subPeak[3] = r_effective
                        
                        if effectiveCoords == True:
                            # Calculate center of overlap
                            xc, yc, zc = intMid(peak_list[0][:,final_halo_index],
                                               peak_list[redshift_index+1][:, index])
                            subPeak[0:3] = xc, yc, zc
                        
                        #print("new peak index: {} with radius {:.3}".format(index, subPeakRadius))
                        #print("at a distance {:.3}".format(dists[i]))
                        new_peaks.append(subPeak)  

            new_peaks = np.asarray(new_peaks)
            if len(new_peaks) == 1:
                peaks[redshift_index+1] = np.vstack(np.asarray(new_peaks))
            elif new_peaks.size>0:
                peaks[redshift_index+1] = np.vstack(np.asarray(new_peaks))
            else:
                peaks[redshift_index+1] = np.asarray([])
            if printOutput == True:
                pass
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

    
def haloOrdering(halo, halolist):
    # Gives the position of a halo in a list based on it's distance from the origin in the x-y plane
    # normalised so that the middle position = 0
    
    r_halo = halo[0]**2 + halo[1]**2
    r_all = np.asarray([halolist[i, 0]**2+halolist[i, 1]**2 for i in range(len(halolist))])
    
    
    posn = len(r_all[r_all<r_halo]) 
    
    norm_posn = posn - len(r_all)/2
    
    return norm_posn


def plotMergerTree(merger_list, ppFile, path_prefix, startIndex=0, printOutput = False, 
                   cmap = 'gnuplot_r', font_size = 15, log = False, colorbar = False, 
                   colorbar_title = None, min_mass = 0, max_mass = None, max_radius=None,
                   figure = None, subplot = None):
    # TODO: Add function description
    # TODO: Sort heights of peaks based on y-axis posn or something.
            
    # Filter out low mass halos
    merger_list = pruneLowMasses(merger_list, min_mass, printOutput)
    
    # only plot for redshifts with peaks in them
    last_index = 0
    for i in range(len(merger_list)-1):
        if merger_list[i].size > 0:
            last_index += 1
    
    ppFile_path = path_prefix + "/inputs/" + ppFile
    
    #print(ppFile_path)
    p = ParamsFile(ppFile_path)
    redshifts = p["redshifts"][startIndex:]
    boxsize = p["boxsize"]
   
    # Move peaks to account for periodic boundary conditions
    merger_list = MoveOutOfBounds(merger_list, boxsize, printOutput)
    
    # plotting info
    matplotlib.rc('font', size=font_size)
    #print(merger_list[-1][:,4]
    # Colour on mass
    colormap = plt.get_cmap(cmap) 
    if max_mass == None:
        max_mass = merger_list[0][0,4]
        
    if max_radius == None:
        max_radius = merger_list[0][0,3]

    #cNorm  = colors.Normalize(np.log10(max(1e-15, min_mass)), np.log10(max_mass))
    #scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
    
    cNorm  = colors.LogNorm(max(1e-16, min_mass), max_mass)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=colormap)
    
    if figure == None:
        fig = plt.figure(figsize = (7, 5))
    else:
        fig = figure
        
    if subplot == None:
        ax1 = fig.add_subplot(111)
    else:
        ax1 = fig.add_subplot(subplot)
    
    # Plot lines between halos
    for i in range(last_index-1, 0, -1):
        if printOutput == True:
            print("redshift index: {}".format(i))
            
        # Check each peak for peaks at the earlier redshift
        for j in range(merger_list[i].shape[0]):
            if i<len(merger_list)-1 and merger_list[i-1].size>0:
                # Calculate distance between current halo and all others at next redshift
                dists = np.sqrt(np.sum((merger_list[i][j,0:3]-merger_list[i-1][:,0:3])**2,axis=1)) -merger_list[i-1][:,3]
                index = np.where(dists == min(dists))[0][0]
                
                assert type(index)==np.int64, "No parent halo found"
                    
                # Plot merger line
                y_posn = [haloOrdering(merger_list[i][j,:], merger_list[i]), 
                          haloOrdering(merger_list[i-1][index,:], merger_list[i-1])]
                ax1.plot([redshifts[i],redshifts[i-1]], y_posn, '--', color = "gray")
        
    # Plot halo
    for i in range(last_index-1, 0, -1):
        if printOutput == True:
            print("redshift index: {}".format(i))
            
        # Check each peak for peaks at the earlier redshift
        for j in range(merger_list[i].shape[0]):           
            # Plot the halo
            ms = 30*merger_list[i][j,3]/max_radius
            colorVal = scalarMap.to_rgba(merger_list[i][j,4])   
            ax1.plot(redshifts[i], haloOrdering(merger_list[i][j,:], merger_list[i]),
                     'o', ms = ms, color = colorVal)    
    
    # Plot final halo
    ms = 30*merger_list[0][0,3]/max_radius
    colorVal = scalarMap.to_rgba(merger_list[0][0,4])     
    ax1.plot(redshifts[0], -0.5, 'o', ms = ms, color = colorVal)    
    
    if log == True:
        plt.xscale('log')
    ax1.set_yticks([])     
    ax1.set_xlim(redshifts[last_index], redshifts[0]*0.8)
    ax1.set_xlabel("Redshift, $z$")
    
    print("earliest redshift = {}".format(redshifts[last_index]))
    #ax1.set_xticks([])
    
    if colorbar == True:
        #scalarMap.set_array([])
        cbar = fig.colorbar(scalarMap)
        if colorbar_title != None:
            assert type(colorbar_title) == str, "Error: colorbar_title must be a string"
            cbar.ax.set_ylabel(colorbar_title)
            
    return ax1

def plotMergerPatches(merger_list, ppFile, path_prefix, printOutput = False, cmap = 'gnuplot'):
    
    last_index = 0
    for i in range(len(merger_list)-1):
        if merger_list[i].size > 0:
            last_index += 1
   
    ppFile_path = path_prefix + "/inputs/" + ppFile
    
    #print(ppFile_path)
    p = ParamsFile(ppFile_path)
    redshifts = p["redshifts"]  
    boxsize = p["boxsize"]
    matplotlib.rc('font', size=15)
    # Move peaks to account for periodic boundary conditions
    merger_list = MoveOutOfBounds(merger_list, boxsize, printOutput)
    
    import matplotlib.colors as colors
    fig = plt.figure(figsize = (11,5))
    colormap = plt.get_cmap(cmap) 
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
        
def FindCollapseRedshift(merger_tree, thresh_frac, pp_file, path_prefix,
                         startIndex = 0, printOutput = False, interp = "None"):
    
    ppFile_path = path_prefix + "/inputs/" + pp_file
    #print(ppFile_path)
    p = ParamsFile(ppFile_path)
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

def getStartIndex(ppFile, path_prefix, z0):
    p = ParamsFile(path_prefix + "/inputs/" + ppFile)
    redshifts = p["redshifts"] 
    nearest_z = redshifts[abs(redshifts-z0)==min(abs(redshifts-z0))][0]
    start_index = len(redshifts[redshifts<nearest_z])
    return start_index