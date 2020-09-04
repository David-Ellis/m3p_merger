import sys, os, m3p_merger
import numpy as np

args = sys.argv

assert len(args) == 3, "2 input arguments required: <pp file name> <save keyword>"

ppFile = args[1]
saveName = args[2]

z0 = 99.0
frac = 0.01   
print("-"*80)
print("Calculating collapse redshifts\n   ppFile: {}\n   Save name = {}\n".format(ppFile, saveName))
    
# Check if output directory exists, if not, make one
if not os.path.isdir("ConcEvolution"):
    print("Output directory \\ConcEvolution does not exit. Make it now.")
    os.makedirs("ConcEvolution")
    
# Generate save names
collapse_save_name = "ConcEvolution/{}_CollapseRedshifts.npy".format(saveName)
masses_save_name = "ConcEvolution/{}_FinalMasses.npy".format(saveName)
radii_save_name = "ConcEvolution/{}_FinalRadii.npy".format(saveName)

# Check that files with these save names don't already exist
abort = False
for name in [collapse_save_name, masses_save_name, radii_save_name]:
    error_message = "Error! File: {} already exists!".format(name)
    if os.path.isfile(name):
        print(error_message)
        abort = True
assert (not abort), "Save name already exists"
    

print("Building peak list...")
peak_list, boxsize = m3p_merger.MakePeakList(ppFile,startIndex = 0, massType = "unstripped", printOutput = True)
print("Done.\n")

print("Building merger trees...")
out = m3p_merger.BuildMergerTree2(peak_list, ppFile, final_halos_indicies = "all", printOutput = True)
print("Done.\n")

print("Calculating collapse redshifts...")
collapse_redshifts = np.zeros(len(out))
for i in range(len(out)):
    collapse_redshifts[i] = m3p_merger.FindCollapseRedshift(out[i], frac, ppFile, interp = "None")[0]
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
