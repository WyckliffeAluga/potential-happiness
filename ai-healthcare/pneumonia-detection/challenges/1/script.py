
import numpy as np
import pandas as pd


# read the file of the current worklist with probabilities that the two algorithms have
# generated for two types of findings most concernes with

worklist = pd.read_csv('probabilities.csv')


# create a new column (time_to_read) showing that every image taking 6 minutes to read will be read in the order they are prepesented
worklist['time_to_read'] = np.arange(6, 6*(len(worklist) + 1), 6)

# create a new column showing the max probability between brain bleed or aortic dissection
worklist['max_prob'] = worklist[['Brain_bleed_probability', 'Aortic_dissection_probability']].max(axis=1)

# reorder the worklist bases on probabilities of critical findings
worklist_prioritized = worklist.sort_values(by=['max_prob'], ascending=True)
worklist_prioritized['time_to_read_prioritized'] = np.arange(6, 6*(len(worklist)+1),6)
worklist_prioritized['time_delta'] = worklist_prioritized['time_to_read'] - worklist_prioritized['time_to_read_prioritized']

# find places where the algorithm saved at least 30 minutes for brain bleeds
worklist_prioritized[((worklist_prioritized.time_delta > 30) & (worklist_prioritized.Image_Type=='head_ct'))]


# do the same for at least 15 minutes with aortic dissections
worklist_prioritized[((worklist_prioritized.time_delta>=15)&(worklist_prioritized.Image_Type=='chest_xray'))]
print(len(worklist_prioritized[((worklist_prioritized.time_delta>=15)&(worklist_prioritized.Image_Type=='chest_xray'))]))

# Look anywhere we missed and props > 0.5 were read slower
worklist_prioritized[((worklist_prioritized.time_delta<0)&(worklist_prioritized.max_prob>=0.5))]
