import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import os
from datetime import date
import sys
import argparse
from pathlib import Path
import json

autolines_path_template = './Hanford/{date_folder}/autolines_annotated_only.txt'
cohmatrix_path_template = './Hanford/{date_folder}/coh_matrix.npz'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            '-i', '--interactive',
            action='store_true',
            help='On/Off flag for the interactive version of this script.')
    
    parser.add_argument(
            '-r', '--range',
            action='extend',
            nargs='+', type=str,
            help='Pass in one or more ISO format dates to be used in getting data files.')

    parser.add_argument(
            '--fmin', type=float,
            default=0,
            help='Minimum frequency that will be put into the heatmap')

    parser.add_argument(
            '--fmax', type=float,
            default=-1,
            help='Maximum frequency that will be put into the heatmap')
    
    parser.add_argument(
            '--show',
            action='store_true',
            help='Display the heatmap after saving it.'
    )

    args = parser.parse_args()

    return args

### Copied from Fscan gitlab
def load_lines_from_linesfile(fname, exclude_tag=None, include_tag=None):
    """
    Load line data from a linesfile. Expects csv data with two entries per row,
    the first being a frequency (float) and the second being a label (which
    cannot include commas)

    Example:

    10.000,First line label
    10.003,Second line label
    ...

    If an .npz file is supplied instead, assume it contains frequencies and
    that there are no labels.

    Parameters
    ----------
    fname : Path, str
        Path to file

    Returns
    -------
    lfreq : 1-d numpy array (dtype: float)
        Array of frequencies
    names : 1-d numpy array (dtype: str)
        strings of names associated with given lines.
    """
    if exclude_tag is None:
        exclude_tag = []
    if include_tag is None:
        include_tag = []

    # Make sure the file path is properly formatted
    fname = Path(fname).expanduser().absolute()

    if fname.suffix == ".npz":
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

    remove = []
    for exclude in exclude_tag:
        for idx, name in enumerate(names):
            if '!' in name and exclude in name.split('!')[1]:
                remove.append(idx)
    lfreq = np.delete(lfreq, remove)
    names = np.delete(names, remove)

    # by default, keep all
    if len(include_tag) != 0:
        remove = []
        for include in include_tag:
            for idx, name in enumerate(names):
                if '!' in name and include not in name.split('!')[1]:
                    remove.append(idx)
        lfreq = np.delete(lfreq, remove)
        names = np.delete(names, remove)

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

def combs_counts_analysis(linespath : str, cohpath : str, fmin : float, fmax : float):
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

    def _get_unique(comb_index : int, temp_index = None):
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
        # Creates two DataFrames out of the coherence and channel data
        coh_df = pd.DataFrame(coh_matrix["cut_coh_table"], columns=None, index=None)
        chan_df = pd.DataFrame(coh_matrix["chanmatrix"], columns=None, index=None)
        
        # Gets the bin indexes for a single comb set based off comb_index
        if temp_index is None:
            bin_index = match_bins(coh_matrix["frequencies"], freqs[combs_dict[combs_list[comb_index]]])
        else:
            bin_index = match_bins(coh_matrix["frequencies"], temp_index)
        
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
    
    def _counts_trim(trim_counts : np.ndarray, trim_chans : np.ndarray, trim_combs : np.ndarray):
        """Removes channel rows with more 0s than 1s and no counts greater than 1.
        """
        
        temp_df = pd.DataFrame(trim_counts.T, trim_chans, trim_combs)
        
        for trim_chan in trim_chans:
            ones_count = 0
            zeros_count = 0
            others_count = 0
            
            for point in temp_df.loc[trim_chan]:
                if point == 1:
                    ones_count += 1
                elif point == 0:
                    zeros_count += 1
                else:
                    others_count += 1
            
            if zeros_count > ones_count and others_count == 0:
                temp_df = temp_df.drop(index=trim_chan)
            else:
                continue

        heatmap_chans = temp_df.index.to_numpy()
        heatmap_counts = temp_df.to_numpy()
        del temp_df
        
        return heatmap_counts, heatmap_chans
    
    count_total, chan_total = _counts_trim(count_total, chan_total, combs_list)
    
    return combs_list, count_total, chan_total

def heatmap(combs : list, counts : np.ndarray, chans : list,  dataset_type : str,
        fmin : float, fmax : float, missing_dates: list[str] | None = None,
        output_name: str | list[str] = "heatmap_name"):
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
    output_name : (str | list[str])
        File name or date range used to name the image that will be generated.
    """
    
    if missing_dates is None:
            missing_dates = []
    
    if np.size(chans) >= 75:
        heatmap_figsize = (28,35)
        text_fontsize = 5
    else:
        heatmap_figsize = (18, 25)
        text_fontsize = 7
    
    fig, ax = plt.subplots(figsize=heatmap_figsize)
    im = ax.imshow(counts)
    ax.set_xticks(np.arange(len(combs)), labels=combs, fontsize=7)
    ax.set_yticks(np.arange(len(chans)), labels=chans, fontsize=7)
    ax.set_aspect(0.6)
    ax.set_adjustable('box')
    plt.xlabel("Comb Freqencies (Hz)", fontsize=10)
    plt.ylabel("Channels", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(length=1, labelsize=7)
    cbar.set_label('Counts', size=10)
    cbar.ax.locator_params(nbins=np.max(counts))

    plt.setp(ax.get_xticklabels(), rotation=90, ha="center",
            rotation_mode="default")
    subtitle = f"{fmin}Hz to {fmax}Hz"
    
    match dataset_type:
        case "day":
            if not isinstance(output_name, str):
                raise TypeError("output_name must be a date string for a single-day heatmap")
            title = date.fromisoformat(output_name)
            log_date = str(title)
            file_name = f"{output_name}_heatmap"
            plt.title(f"Correlation Between Combs And Channels\n{title}\n{subtitle}", fontsize=12, loc='center')
        case "multi-day":
            if not isinstance(output_name, list) or not output_name or not all(isinstance(date_str, str) for date_str in output_name):
                raise TypeError("output_name must be a list of date strings for a multi-day heatmap")
            title_date_start = date.fromisoformat(output_name[0])
            title_date_end = date.fromisoformat(output_name[-1])
            log_date = [str(title_date_start), str(title_date_end)]
            file_name = f"{output_name[0]}_to_{output_name[-1]}_heatmap"
            plt.title(f"Correlation Between Combs And Channels\n{title_date_start} to {title_date_end}\n{subtitle}", fontsize=12, loc='center')
    
    for i in range(np.size(chans)):
        for j in range(len(combs)):
            # Annotates the text as black if the count percentage is greater than
            # or equal to 80%
            if (counts[i,j]/np.max(counts)) * 100 >= 80:
                text = ax.text(j, i, str(int(counts[i,j])),
                            ha="center", va="center", color="k",
                            fontsize=text_fontsize)
            else:
                text = ax.text(j, i, str(int(counts[i,j])),
                            ha="center", va="center", color="w",
                            fontsize=text_fontsize)
    
    out_dpi = 300
    if os.path.isdir('./media/'):
        print('media directory verified')
        pass
    else:
        print('media directory does not exist', '\n',
                'Creating directory in local directory')
        os.mkdir('./media/')
        
    log_dict = {
                "Generation Date" : str(date.today()),
                "Heatmap Date(s)" : log_date,
                "Selected Frequency Range" : subtitle,
                "Selected Combs" : [float(c) for c in combs],
                "Missing Dates" : missing_dates
                }
    print(log_dict)
    print("log_dict type:", type(log_dict))

    ### If you want to save the heatmaps to a different directory, make sure to change the file paths
    ### in your local version. The default directory may be changed later to something more "central".
    fig.savefig(f'./media/{file_name}.png', dpi=out_dpi)
    with open(f'./media/{file_name}_log.json', 'w') as file:
        file.write(json.dumps(log_dict, indent=5))
    file.close()
    print('Heatmap and log file have been generated successfully.')
    
def multi_date_total(date_list : list, fmin : float, fmax : float):
    """Compare data pulled from combs_counts_analysis over multiple days and create a totaled
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
    missing_dates = []

    # print(date_list)
    for curr_date in date_list:
        # print(curr_date)
        # Checks if the folder with the date name exists. Can be updated to work with LIGO cluster folders
        if not os.path.exists(cohmatrix_path_template.format(date_folder=curr_date)):
            print(f"{curr_date} coh_matrix.npz does not exist, continuing to next date...")
            missing_dates.append(curr_date)
            continue
        elif not os.path.exists(autolines_path_template.format(date_folder=curr_date)):
            print(f"{curr_date} autolines_annotated_only.txt does not exist, continuing to next date...")
            missing_dates.append(curr_date)
            continue
        else:
            combs, counts, chans = combs_counts_analysis(
                     autolines_path_template.format(date_folder=curr_date),
                     cohmatrix_path_template.format(date_folder=curr_date),
                     fmin, fmax)

        # Checks if the container DataFrame is currently empty. If so, reinitializes it to the first set of data
        if total_df.size == 0:
            total_df = pd.DataFrame(counts, chans, combs)

        combs, counts, chans = combs_counts_analysis(
                autolines_path_template.format(date_folder=curr_date),
                cohmatrix_path_template.format(date_folder=curr_date),
                fmin, fmax)
            
        # Checks if the container DataFrame is currently empty. If so, reinitializes it to the first set of data
        if total_df.size == 0:
            total_df = pd.DataFrame(counts.T, chans, combs)

            # print("Continue has been reached")
            continue
            
        # Secondary container DataFrame used to compare against total_df

        curr_df = pd.DataFrame(counts, chans, combs)
            
        for curr_comb in combs:
            # Saves the pandas Series that coincides with the current comb into a new variable
            # then trims the series down to only the non-zero points and saves the index of the non-zero
            # array into a new array
            curr_series = curr_df[curr_comb]
            non_zero = curr_series[curr_series != 0]
            non_zero_array = np.array(non_zero)
            non_zero_index = non_zero.index.to_numpy()
            
            if curr_comb in total_df.columns:
                for i in range(non_zero_index.size):
                    # Looping through the length of the non-zero index array, checks if each channel and comb
                    # pair already exist in the totalized DataFrame. If a pair exist and the data point for
                    # that pair is an NA value (aka added from a previous comb), the NA value is replaced
                    # with the current pair value in the current/stored DataFrame.
                    # If a pair does not exist yet, the data point is added into the totalized DataFrame
                    if non_zero_index[i] in total_df.index:
                        if pd.isna(total_df.at[non_zero_index[i], curr_comb]):
                            total_df.at[non_zero_index[i], curr_comb] = non_zero_array[i]
                        else:
                            total_df.at[non_zero_index[i], curr_comb] = (non_zero_array[i] + total_df.at[non_zero_index[i], curr_comb])
                    else:
                        # If a channel does not yet exist in the totalized DataFrame, the channel is added to
                        # the index and a data point is inserted at the current index-comb pair location.
                        total_df.at[non_zero_index[i], curr_comb] = non_zero_array[i]
            else:
                # If an entire comb does not yet exist in the totalized DataFrame, an "empty" (all zeros)
                # column is inserted then backfilled according to index-comb pairs.
                total_df[curr_comb] = np.zeros_like(np.array(total_df)[:,0])
                for i in range(non_zero_index.size):
                    total_df.at[non_zero_index[i], curr_comb] = non_zero_array[i]
        
        # Sorts combs from smallest to largest, then columbs from A-Z (at least, as close as it can get with
        # auxillary channel names)
        total_df = total_df[sorted(total_df.columns)]
        total_df = total_df.loc[sorted(total_df.index)]
        
        # Converts the DataFrame into a numpy array and replaces all remaining NA values with 0
        # Channel and comb arrays are also created from the index and columns lists.
        total_counts = total_df.to_numpy(na_value=0)
        total_chans = total_df.index.to_list()
        total_combs = total_df.columns.to_list()

    return total_combs, total_counts, total_chans, missing_dates

def interactive_prompts(fmin : float, fmax : float):
    questions = [
    inquirer.List(
        "range",
        message="Select a date range",
        choices=["Single Day", "Multiple Days"],
        ),
    ]
        
    answers = inquirer.prompt(questions)
    if not answers:
        return

    match answers.get("range"):
        case "Single Day":
            date_folder = input("Type in the date in ISO format: ")
            combs, counts, chans = combs_counts_analysis(
                    autolines_path_template.format(date_folder=date_folder),
                    cohmatrix_path_template.format(date_folder=date_folder),
                    fmin, fmax)
            heatmap(combs, counts, chans,
                    'day',
                    fmin, fmax,
                    output_name=date_folder)
                    
        case "Multiple Days":
            dates = input("Type in a start date and end date in ISO format separated by a dash (XX-XX): ")
            dates = dates.split("-")
            
            # Generates a list of dates from the first date to the last via pandas' date_range function
            file_list = [
                d.strftime("%Y%m%d")
                for d in pd.date_range(dates[0], dates[1]).to_list()
            ]
            
            # print(file_list)
            total_combs, total_counts, total_chans, missing_dates = multi_date_total(file_list, fmin, fmax)
            heatmap(total_combs, total_counts, total_chans,
                    'multi-day',
                    fmin, fmax, missing_dates,
                    file_list)

def main(args=None):
    
    if args is None:
        args = get_args()
        #print(args)

    if args.interactive == True:
        try:
            import inquirer
        except (ImportError, NotImplementedError):
            raise SystemExit("Interactive mode needs a real terminal. Use -r instead, e.g. -r 20231230")
        interactive_prompts(args.fmin, args.fmax)
    elif args.range is None:
        raise SystemExit("Pass one date (-r 20231230) or a start and end date (-r 20231225 20231230), or use -i")
    else:
        if len(args.range) == 1:
            combs, counts, chans = combs_counts_analysis(
                    autolines_path_template.format(date_folder=args.range[0]),
                    cohmatrix_path_template.format(date_folder=args.range[0]),
                    args.fmin, args.fmax)
            heatmap(combs, counts, chans,
                    'day',
                    args.fmin, args.fmax,
                    output_name = args.range[0])
        else:
            file_list = pd.date_range(args.range[0], args.range[1]).strftime("%Y%m%d").tolist()

            total_combs, total_counts, total_chans, missing_dates = multi_date_total(file_list,
                    args.fmin, args.fmax)
            heatmap(total_combs, total_counts, total_chans,
                    'multi-day',
                    args.fmin, args.fmax, missing_dates,
                    file_list)
    
    if args.show:
        plt.show()
    
if __name__ == "__main__":
    main()
