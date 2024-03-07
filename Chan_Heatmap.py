import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

### Copied from Fscan gitlab
def load_lines_from_linesfile(fname):
    '''
    Load line data from a linesfile. Expects csv data with two entries per row,
    the first being a frequency (float) and the second being a label (which
    cannot include commas)

    Example:

    10.000,First line label
    10.003,Second line label
    ...

    If an .npy file is supplied instead, assume it contains frequencies and
    that there are no labels.

    Parameters
    ----------
    fname: string
        Path to file

    Returns
    -------
    lfreq: 1-d numpy array (dtype: float)
        Array of frequencies
    names: 1-d numpy array (dtype: str)
        strings of names associated with given lines.
    '''

    # Make sure the file path is properly formatted
    fname = os.path.abspath(os.path.expanduser(fname))

    if fname.endswith(".npy"):
        lfreq = np.load(fname)
        names = np.array([""]*len(lfreq))
    else:
        # Load the data
        linesdata = np.genfromtxt(fname, delimiter=",", dtype=str)
        if len(linesdata) == 0:
            print("Linesfile does not contain any data.")
            return [], []
        lfreq = linesdata[:, 0].astype(float)
        names = linesdata[:, 1]

    return lfreq, names

### Copied from Fscan gitlab
def match_bins(spect, marks):
    ''' For some set of artifact/line frequencies (marks), find
    the indices of the closest frequency bin centers in a spectrum (spect).

    Parameters
    ----------
    spect: 1d array (dtype: float)
        spectral bin center frequencies

    marks: 1d array (dtype: float)
        artifact/line frequencies

    Returns
    -------
    inds: 1d array (dtype: integer)
        indices of spectral bin centers nearest to marks
    '''

    if len(marks) == 0:
        return np.array([])
    # for each bincenter, figure out the distance to next bincenter
    binwidths = np.diff(spect)

    # the rightmost bincenter nothing after it, so use the preceding binwidth
    binwidths = np.append(binwidths, binwidths[-1])
    # for each bin center, the right edge of the bin should be 1/2 of the
    # rightward binwidth
    edges = spect + binwidths/2.
    # the leftmost bin has no left edge so use the subsequent binwidth
    edges = np.append(spect[0]-binwidths[0]/2., edges)

    if min(marks) < edges[0]:
        edges[0] = min(marks)-.1
    if max(marks) > edges[-1]:
        edges[-1] = max(marks)+.1

    # now digitize, using the calculated bin edges
    inds = np.digitize(marks, edges)

    # subtract 1 off the results so that they correspond appropriately to the
    # original bincenters
    inds -= 1

    # raise an exception if we got any results that aren't within the spectrum
    # bounds (returning negative numbers will create unexpected results)

    if len(inds) > 0:
        if np.amin(inds) < 0 or np.amax(inds) > len(spect)-1:
            raise Exception(
                "Not all tested values are within the spectrum bounds.")

    return inds

linespath = "autolines_annotated_only.txt"
cohpath = "coh_matrix.npz"

# Loads coh_maxtrix.npz into a temporary variable
test = np.load(cohpath)
freqs, line_names = load_lines_from_linesfile(linespath)

# Splits the line_names strings and checks the corresponding
# element for a comb frequency. Comb frequencies appended
# onto combs_list
combs_list = []
for i in range(len(line_names)):
    testname = line_names[i].split()
    comb = float(testname[5].split(';')[0])
    combs_list.append(comb)
combs_list = np.unique(combs_list)

# Places all 
combs_dict = {}
for comb in range(len(combs_list)):
    line_index = []
    for index in range(len(line_names)):
        if str(combs_list[comb]) in line_names[index]:
            line_index.append(index)
    combs_dict.update({combs_list[comb] : line_index})

# Creates two DataFrames out of the coherence and channel data
coh_df = pd.DataFrame(test["cut_coh_table"], columns=None, index=None)
chan_df = pd.DataFrame(test["chanmatrix"], columns=None, index=None)

def get_unique(comb_index : int):
    ''' Finds and returns an array of unique channels and
    the number of times those channels were counted.
    
    Parameters
        comb_index : int
            Index number for which comb is being used
    
    Returns
        curr_chan_uq : 1darray
            A 1darray of the unique channels that captured the
            frequencies of the current comb
        
        curr_chan_count : 1darray
            A 1darray filled with the counts of each corresponding
            channel in curr_chan_uq
    '''
    # Gets the bin indexes for a single comb set based off comb_index
    bin_index = match_bins(test["frequencies"], freqs[combs_dict[combs_list[comb_index]]])
    
    # Splits coh_df and chan_df index-wise by bin_index
    coh_arr = np.array(coh_df.loc[sorted(bin_index)])
    chan_arr = np.array(chan_df.loc[sorted(bin_index)])

    # Returns an array containing only channels where
    # their corresponding coherence value is over 0.05
    chan_arr_sort = chan_arr[coh_arr > 0.05]

    # Returns two arrays containing the unique channels found as well as
    # their counts
    curr_chan_uq, curr_chan_count = np.unique(chan_arr_sort, return_counts=True)
    
    return curr_chan_uq, curr_chan_count

# Creates a 1d array that stacks all unique channels together
chan_total = np.array([])
for comb in range(len(combs_list)):
    chan_uq, chan_count = get_unique(comb)
    chan_total = np.append(chan_total, chan_uq)

# Resets chan_total to only the unique channels to remove repeats
chan_total = np.unique(chan_total)

# Pre-initializes an empty 2darray set to the size of combs_list x chan_total
count_total = np.empty((np.size(combs_list), np.size(chan_total)))

# After getting the unique channels, runs through chan_total and checks against
# chan_uq for each unique channel. If one is missing from chan_uq, inserts
# a 0 in chan_count into the corresponding element
for comb in range(len(combs_list)):
    chan_uq, chan_count = get_unique(comb)
    
    for i in range(len(chan_total)):
        if np.isin(chan_total[i], chan_uq, assume_unique=True):
            pass
        else:
            chan_count = np.insert(chan_count, i, 0)

    # After the above check, places the counts array into count_total at
    # the corresponding comb index
    count_total[comb] = chan_count
del coh_df, chan_df

fig, ax = plt.subplots(figsize=(6,25))
im = ax.imshow(count_total.T)
ax.set_xticks(np.arange(len(combs_list)), labels=combs_list, fontsize=7)
ax.set_yticks(np.arange(len(chan_total)), labels=chan_total, fontsize=7)
plt.xlabel("Comb Freqencies (Hz)", fontsize=7)
plt.ylabel("Channels", fontsize=7)
cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.ax.tick_params(length=1, labelsize=7)
cbar.set_label('Counts', size=7)
cbar.ax.locator_params(nbins=np.max(count_total))

plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
         rotation_mode="default")

for i in range(len(chan_total)):
    for j in range(len(combs_list)):
        # Annotates the text as black if the count percentage is greater than
        # or equal to 80%
        if (count_total.T[i,j]/np.max(count_total)) * 100 >= 80:
            text = ax.text(j, i, int(count_total.T[i,j]),
                           ha="center", va="center", color="k",
                           fontsize=7)
        else:
            text = ax.text(j, i, int(count_total.T[i,j]),
                           ha="center", va="center", color="w",
                           fontsize=7)
plt.title("Correlation Between Combs And Channels\nDecember 31st 2023", fontsize=7, loc='center')
fig.savefig('Combs_Channels_Heatmap2.png', dpi=500)
plt.show()
