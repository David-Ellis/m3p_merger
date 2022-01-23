import os
from m3p_merger import m3p_merger
import numpy as np

# Define threshold progenitor mass fraction
frac = 0.01   

# Define m3p file locations
m3p_data_path = r"C:\Users\david\AxionData\PeakPatch\m3p_merger"
ppFile = "inputs.ax_jan5_stitched"
saveName = "jan5_merger"
####################################################################################################

print("-"*80)
print("Calculating collapse redshifts\n   ppFile: {}\n   Save name = {}\n".format(ppFile, saveName))
    
# Check if output directory exists, if not, make one
outfile = "mergerEvolution"

if not os.path.isdir(outfile):
    print("Output directory \\ConcEvolution does not exit. Make it now.")
    os.makedirs(outfile)
    
# Generate save names
collapse_save_name = outfile + "/{}_CollapseRedshifts.npy".format(saveName)
masses_save_name   = outfile + "/{}_FinalMasses.npy".format(saveName)
radii_save_name    = outfile + "/{}_FinalRadii.npy".format(saveName)

# Check that files with these save names don't already exist
abort = False
for name in [collapse_save_name, masses_save_name, radii_save_name]:
    error_message = "Error! File: {} already exists!".format(name)
    if os.path.isfile(name):
        print(error_message)
        abort = True
assert (not abort), "Save name already exists"

print("Building peak list...")
peak_list, boxsize = m3p_merger.MakePeakList(ppFile, m3p_data_path, startIndex = 0, massType = "unstripped", printOutput = True)
print("Done.\n")

print("Building merger trees...")
out = m3p_merger.BuildMergerTree(peak_list, ppFile, m3p_data_path,  final_halos_indicies = "all", printOutput = True)
print("Done.\n")

print("Calculating collapse redshifts...")
collapse_redshifts = np.zeros(len(out))
for i in range(len(out)):
    collapse_redshifts[i] = m3p_merger.FindCollapseRedshift(out[i], frac, ppFile, m3p_data_path, interp = "None")[0]
print("Done.\n")

print("Fetching final halo masses...")
masses = np.zeros(len(out))
for i in range(len(out)):
    masses[i] = out[i][0][0,4]
print("Done.\n")

print("Fetching final halo radii...")
radii = np.zeros(len(out))
for i in range(len(out)):
    radii[i] = out[i][0][0,3]
print("Done.\n")

print("Saving data...")
np.save(collapse_save_name, collapse_redshifts)
np.save(masses_save_name, masses)
np.save(radii_save_name, radii)
print("Done.\n")
