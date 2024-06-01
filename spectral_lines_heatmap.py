import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import os
from datetime import date

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

def combs_counts_analysis(linespath : str, cohpath : str):
    """Analyze combs, channel, and coherence data passed in
    by linespath and cohpath. Returns an array of combs, an
    array of channel counts, and a list of all unique channels
    with a coherence value of 0.05 or greater.

    Parameters
    ----------
    linespath : str
        The file path for autolines_annotated_only.txt
    cohpath : str
        The file path for coh_matrix.npz

    Returns
    -------
    combs_list : NDArray
        An array of combs found within the passed in data
    chan_total : NDArray
        An array containing every unique channel with a coherence value
        of 0.05 or greater
    count_total : 2DArray
        A 2D array containing the counts of every channel for each comb.
        The array is of size combs_list x  chan_total
    """

    if not os.path.exists(linespath):
        print(f"{linespath} folder does not exist, stopping program...")
        sys.exit()

    # Loads coh_maxtrix.npz into a temporary variable
    coh_matrix = np.load(cohpath)
    freqs, line_names = load_lines_from_linesfile(linespath)

    # Splits the line_names strings and checks the corresponding
    # element for a comb frequency. Comb frequencies appended
    # onto combs_list
    combs_list = []
    for i in range(len(line_names)):
        comb_w_offset = line_names[i].split()
        comb = float(comb_w_offset[-4].split(';')[0])
        combs_list.append(comb)
    combs_list = np.unique(combs_list)

    # Places all frequencies and their respective indexes into a dictionary relative
    # to which comb frequency they are a multiple of
    combs_dict = {}
    for comb in range(len(combs_list)):
        line_index = []
        for index in range(len(line_names)):
            if str(combs_list[comb]) in line_names[index]:
                line_index.append(index)
        combs_dict.update({combs_list[comb] : line_index})
    
    # Creates two DataFrames out of the coherence and channel data
    coh_df = pd.DataFrame(coh_matrix["cut_coh_table"], columns=None, index=None)
    chan_df = pd.DataFrame(coh_matrix["chanmatrix"], columns=None, index=None)

    def _get_unique(comb_index : int):
        ''' Finds and returns an array of unique channels and
        the number of times those channels were counted.
        
        Parameters
        ----------
        comb_index : int
            Index number for which comb is being used
        
        Returns
        -------
        curr_chan_uq : 1darray
            A 1darray of the unique channels that captured the
            frequencies of the current comb
        
        curr_chan_count : 1darray
            A 1darray filled with the counts of each corresponding
            channel in curr_chan_uq
        '''
        # Gets the bin indexes for a single comb set based off comb_index
        bin_index = match_bins(coh_matrix["frequencies"], freqs[combs_dict[combs_list[comb_index]]])
        
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
        chan_uq, chan_count = _get_unique(comb)
        chan_total = np.append(chan_total, chan_uq)

    # Resets chan_total to only the unique channels to remove repeats
    chan_total = np.unique(chan_total)

    # Pre-initializes an empty 2darray set to the size of combs_list x chan_total
    count_total = np.empty((np.size(combs_list), np.size(chan_total)))

    # After getting the unique channels, runs through chan_total and checks against
    # chan_uq for each unique channel. If one is missing from chan_uq, inserts
    # a 0 in chan_count into the corresponding element
    for comb in range(len(combs_list)):
        chan_uq, chan_count = _get_unique(comb)
        
        for i in range(len(chan_total)):
            if np.isin(chan_total[i], chan_uq, assume_unique=True):
                pass
            else:
                chan_count = np.insert(chan_count, i, 0)

        # After the above check, places the counts array into count_total at
        # the corresponding comb index
        count_total[comb] = chan_count
    del coh_df, chan_df
    
    return combs_list, count_total, chan_total

def heatmap(combs : np.ndarray, counts : np.ndarray, chans : np.ndarray,  dataset_type : str,
            output_name : str | list = "heatmap_name"):
    """Generate a matplotlib heatmap from data gotten from combs_counts_analysis.

    Parameters
    ----------
    combs : (np.ndarray)
        An array of combs found within the passed in data
    counts : (np.ndarray)
        A 2D array containing the counts of every channel for each comb.
        The array is of size combs_list x  chan_total
    chans : (np.ndarray)
        An array containing every unique channel with a coherence value
        of 0.05 or greater
    output_name : (str)
        File name for the image that will be generated
    """
    fig, ax = plt.subplots(figsize=(6,20))
    im = ax.imshow(counts)
    ax.set_xticks(np.arange(len(combs)), labels=combs, fontsize=7)
    ax.set_yticks(np.arange(len(chans)), labels=chans, fontsize=7)
    plt.xlabel("Comb Freqencies (Hz)", fontsize=10)
    plt.ylabel("Channels", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(length=1, labelsize=7)
    cbar.set_label('Counts', size=10)
    cbar.ax.locator_params(nbins=np.max(counts))

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
            rotation_mode="default")

    if np.max(counts) > 100:
        text_fontsize = 3
        out_dpi = 300
    else:
        text_fontsize = 5
        out_dpi = 200
    
    match dataset_type:
        case "day":
            title = date.fromisoformat(output_name)
            file_name = f"{output_name}_heatmap"
            plt.title(f"Correlation Between Combs And Channels\n{title}", fontsize=12, loc='center')
        case "multi-day":
            text_fontsize = 3
            out_dpi = 300
            title_date_start = date.fromisoformat(output_name[0])
            title_date_end = date.fromisoformat(output_name[-1])
            file_name = f"{output_name[0]}_to_{output_name[-1]}_heatmap"
            plt.title(f"Correlation Between Combs And Channels\n{title_date_start} to {title_date_end}", fontsize=12, loc='center')
        case "week":
            title = date.fromisoformat(output_name)
            file_name = f"{output_name}_heatmap"
            plt.title(f"Correlation Between Combs And Channels\nWeek of {title}", fontsize=10, loc='center')
        case "multi week":
            title_date_start = date.fromisoformat(output_name[0])
            title_date_end = date.fromisoformat(output_name[-1])
            file_name = f"{output_name[0]}_to_{output_name[-1]}_heatmap"
            plt.title(f"Correlation Between Combs And Channels\nWeeks of {title_date_start} to {title_date_end}", fontsize=10, loc='center')
    
    for i in range(np.size(chans)):
        for j in range(len(combs)):
            # Annotates the text as black if the count percentage is greater than
            # or equal to 80%
            if (counts[i,j]/np.max(counts)) * 100 >= 80:
                text = ax.text(j, i, (counts[i,j]),
                            ha="center", va="center", color="k",
                            fontsize=text_fontsize)
            else:
                text = ax.text(j, i, (counts[i,j]),
                            ha="center", va="center", color="w",
                            fontsize=text_fontsize)
            
    fig.savefig(f'{file_name}.png', dpi=out_dpi)
    plt.show()
    
if __name__ == "__main__":
    import inquirer
    import sys
    
    # Work in progress addition to main module. Will eventually allow users to select between multiple types of date ranges
    # and average out multiple days if selected
    # For testing run either in this file itself or in a command line (Powershell or Command Prompt window)
    def multi_date_total(date_list : list):
        """Compare data pulled from combs_counts_analysis over multiple days and create an "averaged"
        set of arrays to be used in heatmaps

        Parameters
        ----------
        date_list : list
            A list of ISO format dates generated from user input

        Returns
        -------
        total_combs: ndarray
            A compiled list of all combs found in all sets of data that have been read
        
        total_counts : ndarray
            A compiled array of all counts found in all sets of data that have been read
        
        total_chans : ndarray
            A compiled list of all channels found in all sets of data that have been read
        """
        # A container DataFrame that will be used to compile all sets of data
        total_df = pd.DataFrame()
        
        for curr_date in date_list:
            print(curr_date)
            # Checks if the folder with the date name exists. Can be updated to work with LIGO cluster folders
            if not os.path.exists(f"Hanford\\{curr_date}"):
                print(f"Hanford\\{curr_date} folder does not exist, continuing to next date...")
                continue
            combs, counts, chans = combs_counts_analysis(f'Hanford\\{curr_date}\\autolines_annotated_only.txt', f'Hanford\\{curr_date}\\coh_matrix.npz')
            
            # Checks if the container DataFrame is currently empty. If so, reinitializes it to the first set of data
            if total_df.size == 0:
                total_df = pd.DataFrame(counts.T, chans, combs)
                continue
            
            # Secondary container DataFrame used to compare against total_df
            curr_df = pd.DataFrame(counts.T, chans, combs)
            
            for curr_comb in combs:
                curr_series = curr_df[curr_comb]
                non_zero = curr_series[curr_series != 0]
                non_zero_array = np.array(non_zero)
                non_zero_index = non_zero.index.to_numpy()
                
                if curr_comb in total_df.columns:
                    for i in range(non_zero_index.size):
                        if non_zero_index[i] in total_df.index:
                            if pd.isna(total_df.at[non_zero_index[i], curr_comb]):
                                total_df.at[non_zero_index[i], curr_comb] = non_zero_array[i]
                            else:
                                total_df.at[non_zero_index[i], curr_comb] = (non_zero_array[i] + total_df.at[non_zero_index[i], curr_comb])
                        else:
                            total_df.at[non_zero_index[i], curr_comb] = non_zero_array[i]
                else: 
                    total_df[curr_comb] = np.zeros_like(np.array(total_df)[:,0])
                    for i in range(non_zero_index.size):
                        total_df.at[non_zero_index[i], curr_comb] = non_zero_array[i]

            total_df = total_df[sorted(total_df.columns)]
            total_counts = total_df.to_numpy(na_value=0)
            total_chans = total_df.index.to_numpy()
            total_combs = total_df.columns.to_numpy()
            
            return total_combs, total_counts, total_chans
    
    questions = [
    inquirer.List(
        "range",
        message="Select a date range",
        choices=["Single Day", "Multiple Days", "Single Week", "Multiple Weeks",
                 "Single Month", "Multiple Months"],
        ),
    ]
        
    answers = inquirer.prompt(questions)
    match answers["range"]:
        
        # Runs the functions as normal and expected. No bugs as far as I'm aware
        # aside from some fringe issues when the in-built dependents are changed
        case "Single Day":
            date_folder = input("Type in the date in ISO format: ")
            combs, counts, chans = combs_counts_analysis(f'Hanford\\{date_folder}\\autolines_annotated_only.txt', f'Hanford\\{date_folder}\\coh_matrix.npz')
            heatmap(combs, counts.T, chans, 'day', date_folder)
        
        # Gave a heatmap once, hasn't worked since. Thinking I need to rework it to
        # use dataframes rather than arrays both for sorting and inserting properly
        case "Multiple Days":
            dates = input("Type in a start date and end date in ISO format separated by a dash (XX-XX): ")
            dates = dates.split("-")
            file_list = pd.date_range(dates[0], dates[1]).to_list()
            for i in range(len(file_list)):
                file_list[i] = file_list[i].strftime("%Y%m%d")
            
            print(file_list)
            total_combs, total_counts, total_chans = multi_date_total(file_list)
            heatmap(total_combs, total_counts, total_chans, 'multi-day', file_list)
        
        case "Single Week":
            sys.exit("WIP Section")
            date_folder = input("Type in the date in ISO format: ")
            combs, counts, chans = combs_counts_analysis(f'{date_folder}\\autolines_annotated_only.txt', f'{date_folder}\\coh_matrix.npz')
            heatmap(combs, counts.T, chans, 'day', date_folder)
        
        case "Multiple Weeks":
            sys.exit("WIP Section")
            dates = input("Type in a start date and end date in ISO format separated by a dash (XX-XX): ")
            dates = dates.split("-")
            file_list = pd.date_range(dates[0], dates[1], freq="W").to_list()
            for i in range(len(file_list)):
                file_list[i] = file_list[i].strftime("%Y%m%d")
            
            total_combs, total_counts, total_chans = multi_date_total(file_list)
            heatmap(total_combs, total_counts, total_chans, file_list)
        
        case "Single Month":
            sys.exit("WIP Section")
            date_folder = input("Type in the date in ISO format: ")
            combs, counts, chans = combs_counts_analysis(f'{date_folder}\\autolines_annotated_only.txt', f'{date_folder}\\coh_matrix.npz')
            heatmap(combs, counts.T, chans, date_folder)
        
        case "Multiple Months":
            sys.exit("WIP Section")
            dates = input("Type in a start date and end date in ISO format separated by a dash (XX-XX): ")
            dates = dates.split("-")
            file_list = pd.date_range(dates[0], dates[1], freq="MS").to_list()
            for i in range(len(file_list)):
                file_list[i] = file_list[i].strftime("%Y%m%d")
            
            total_combs, total_counts, total_chans = multi_date_total(file_list)
            heatmap(total_combs, total_counts, total_chans, file_list)
