'''
CalcColl.py

Basic script to calculate and save all merger trees for a given m3p data set.
'''

import numpy as np
import m3p_merger

# Peak patch data
ppFile = "/usr/users/ellis/PeakPatch/m3p/inputs/inputs.ax_manyz"
print("Calculating collapse redshifts for final halos of m3p run: {}\n".format(ppFile))

print("Building peak list...")
peak_list, boxsize = m3p_merger.MakePeakList(ppFile, printOutput = True)
print("Done.\n")

print("Building merger trees...")
out = m3p_merger.BuildMergerTree(peak_list, ppFile, final_halos_indicies = "all", printOutput = True)
print("Done.\n")

print("Calculating collapse redshifts...")
collapse_redshifts = np.zeros(len(out))
for i in range(len(out)):
    collapse_redshifts[i] = m3p_merger.FindCollapseRedshift(out[i], 0.01, ppFile)
print("Done.\n")

print("Calculating final halo masses...")
masses = np.zeros(len(out))
for i in range(len(out)):
    masses[i] = out[i][0][0,4]
print("Done.\n")

print("Saving data...")
np.save("data/ax_manyz_CollapseRedshifts_f1em2", collapse_redshifts)
np.save("data/ax_manyz_FinalMasses_f1em2", masses)
print("Done.\n")
