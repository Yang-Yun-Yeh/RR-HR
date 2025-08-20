from utils.visualize import *
from utils.signal_process import *
from utils.preprocess import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime

import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description="Label  GT peaks")

    # Path
    parser.add_argument('-f', '--data_folder', type=str, default="data/all", help='path to the .csv files folder needed to be labelled')
    parser.add_argument("-d", "--file_path", type=str, default='dataset/gt', help='directory to save labelled gt .pkl file')
    parser.add_argument("-n", "--file_name", type=str, default='all', help='.pkl name save labelled gt')

    # Dataset
    parser.add_argument('--p', nargs='+', default=['hamham', 'kiri', 'engineer',
                                                   'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 
                                                   'w1', 'w2', 'w3', 'w4', 'w5']
                                                   , help='specify people needed to be labelled')
    parser.add_argument('--a', nargs='+', default=['walk', 'run'], help='specify actions needed to be labelled')
    
    # Overwrite
    parser.add_argument("--overwrite", action='store_true', help='overwrite files which had been labelled before')


    args = parser.parse_args()
    return args

class PeakLabeler:
    """
    An interactive plot to manually label peaks in a time-series dataset.

    This class creates a matplotlib figure with two subplots:
    1. A static overview plot showing the entire dataset and the current view.
    2. A main plot for detailed viewing, scrolling, and labeling.
    """
    def __init__(self, file_path, person='', file_name='', window_size=30.0):
        """
        Initializes the PeakLabeler application.

        Args:
            file_path (str): The path to the CSV file containing the data.
                            The CSV should have 'second' and 'force' columns.
            window_size (float): The time window (in seconds) for the main plot.
        """
        # --- 1. Data Loading and Initialization ---
        self.window_size = window_size
        self.peaks = []      # List to store the x-coordinates (time) of labeled peaks
        self.indices = []    # List to store the x-coordinates (index) of labeled peaks
        self.markers = []    # List to store tuples of (main_marker, overview_marker)

        self.e = False # exit & not save
        self.q = False # exit & save
        self.n = False # skip current file

        # Load data using pandas
        try:
            self.data = pd.read_csv(file_path)
            self.force = self.data['Force'].values
            self.file_len = self.data.shape[0]

            # compute time from Timestamp
            timestamp_strings = self.data['Timestamp'].values
            datetime_objects = [datetime.fromisoformat(ts) for ts in timestamp_strings]
            start_time = datetime_objects[0]
            self.time = np.array([(dt - start_time).total_seconds() for dt in datetime_objects])

        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return

        # --- 2. Plot Setup with Two Subplots ---
        # Create a figure with two subplots stacked vertically.
        # **FIX**: Removed `sharex=True` to allow independent x-axes.
        self.fig, (self.ax_overview, self.ax_main) = plt.subplots(
            2, 1, 
            gridspec_kw={'height_ratios': [1, 3]} # Make top plot 1/3 height of bottom
        )
        self.fig.suptitle(f'Subject:{person}, {file_name}', fontsize=16) # Peak Labeling Tool
        plt.subplots_adjust(bottom=0.2, hspace=0.3) # Adjust layout for slider and title

        # --- Setup for Top (Overview) Plot ---
        self.ax_overview.plot(self.time, self.force, color='gray', linewidth=0.8)
        self.ax_overview.set_title('Overall Data View')
        # **FIX**: Explicitly set the x-limits to the full data range.
        self.ax_overview.set_xlim(self.time.min(), self.time.max())
        self.ax_overview.set_yticklabels([]) # Hide y-axis labels for a cleaner look
        self.ax_overview.grid(True, linestyle='--', alpha=0.6)
        
        # Add a shaded region to indicate the current view of the main plot
        # self.ax_overview_white = self.ax_overview.copy()
        self.view_span = self.ax_overview.axvspan(
            0, self.window_size, color='blue', alpha=0.3, zorder=-1
        )

        # Add a vertical line to the overview plot to track the mouse cursor
        self.cursor_line = self.ax_overview.axvline(self.time[0], color='r', linestyle='--', linewidth=1, visible=False)

        # --- Setup for Bottom (Main) Plot ---
        self.ax_main.plot(self.time, self.force, color='green')
        self.ax_main.set_title('Label Peaks')
        self.ax_main.set_xlabel('second')
        self.ax_main.set_ylabel('RR force')
        self.ax_main.grid(True)
        self.ax_main.set_xlim(0, self.window_size)
        # Set y-limits based on the entire dataset for consistency
        self.ax_main.set_ylim(self.force.min() - 1, self.force.max() + 1)
        self.ax_overview.set_ylim(self.force.min() - 1, self.force.max() + 1)

        # Add a text box at the bottom of the figure for instructions.
        manual_text = (
            'Left Click: Label Peak  |  Right Click: Undo Last Label\n'
            '[c]: Save & Next  |  [n]: Skip File  |  [q]: Exit & Save All  |  [e]: Exit & Discard All'
        )
        self.fig.text(0.5, 0.03, manual_text, ha='center', va='bottom', 
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="wheat", alpha=0.5))

        # --- 3. Scrollbar (Slider) Widget ---
        ax_slider = plt.axes([0.20, 0.08, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        max_time = self.time.max() - self.window_size
        self.slider = Slider(ax=ax_slider, label='Time', valmin=0, valmax=max_time, valinit=0)
        self.slider.on_changed(self.update_plot_view)

        # --- 4. Connect Matplotlib Events ---
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)

    def peaks_to_indices(self, peaks):
        indices = [-1] * len(peaks)

        # peaks (time) -> sorted
        for i, p in enumerate(peaks):
            idx = (np.abs(self.time - p)).argmin()
            indices[i] = idx

        return indices

    def update_plot_view(self, val):
        """Callback to update the plot when the slider is moved."""
        start_time = self.slider.val
        end_time = start_time + self.window_size
        
        # Update the x-axis limit of the main plot
        self.ax_main.set_xlim(start_time, end_time)
        
        # Update the position of the shaded region in the overview plot
        self.view_span.set_x(start_time)
        
        self.fig.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse motion events to show cursor position on overview."""
        # Check if the mouse is inside the main plot area and has valid data coords
        if event.inaxes == self.ax_main and event.xdata is not None:
            # Update the vertical line's position and make it visible
            # print('condition')
            self.cursor_line.set_xdata([event.xdata, event.xdata])
            self.cursor_line.set_visible(True)
        else:
            # If the mouse is outside, make the line invisible
            self.cursor_line.set_visible(False)
        
        # print('on_mouse_move')
        # Redraw the canvas to show the updated line
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        """Callback for mouse click events."""
        # We only care about left-clicks on the main plot
        if event.button == 1:
            if event.inaxes != self.ax_main: # or event.button != 1:
                return
                
            x_coord, y_coord = event.xdata, event.ydata
            self.peaks.append(x_coord)
            
            # Add a marker to BOTH plots
            main_marker = self.ax_main.plot(x_coord, y_coord, 'rx', markersize=10)[0]
            overview_marker = self.ax_overview.plot(x_coord, y_coord, 'rx', markersize=5)[0]
            self.markers.append((main_marker, overview_marker))
            
            print(f"Peak added at {x_coord:.2f} seconds. Current peaks: {len(self.peaks)}")
            self.fig.canvas.draw_idle()

        # right click: remove the most recent label
        if event.button == 3:
            if self.markers:
                last_peak = self.peaks.pop()
                
                # Remove the markers from BOTH plots
                last_main_marker, last_overview_marker = self.markers.pop()
                last_main_marker.remove()
                last_overview_marker.remove()
                
                print(f"Removed peak at {last_peak:.2f} seconds. Peaks remaining: {len(self.peaks)}")
                self.fig.canvas.draw_idle()
            else:
                print("No peaks to remove.")

    def on_key(self, event):
        """Callback for key press events."""
        # 'c' key: finish labeling, sort data, and close; 'q' key: exit & save
        if event.key == 'c' or event.key == 'q':
            self.peaks.sort()
            print("\n--- Labeling Complete ---")
            print(f"Final number of peaks: {len(self.peaks)}")
            final_peaks_str = ", ".join([f"{p:.2f}" for p in self.peaks])
            self.indices = self.peaks_to_indices(self.peaks)
            print(f"Sorted peak times (s): [{final_peaks_str}]")
            print(f"Sorted peak indices: {self.indices}")

            if event.key == 'q':
                self.q = True

            plt.close(self.fig)

        # 'e' key: not save & exit
        if event.key == 'e':
            self.e = True
            plt.close(self.fig)

        # 'n' key: skip current file
        if event.key == 'n':
            self.n = True
            plt.close(self.fig)

if __name__ == '__main__':
    # person = 'm7'
    # action_name = "walk_0621_0422"
    # file_path = f'./data/all/{person}/{action_name}.csv'

    args = parse_args()
    e, q, save = False, False, False
    gt = {}

    # Check whether load existing file
    labelled_files = []
    if os.path.exists(f'{args.file_path}/{args.file_name}.pkl'):
        print(f"The file '{args.file_path}/{args.file_name}.pkl' exists.")
        gt = pickle.load(open(f'{args.file_path}/{args.file_name}.pkl', 'rb'))
        
        if not args.overwrite:
            labelled_files = list(gt.keys())

        # people = [key for key in gt]
        # for i, person in enumerate(people):
        #     for j, file in enumerate(gt[person]):
        #         file_name = file['file_name']
        #         if not args.overwrite:
        #             labelled_files.append(file_name)
    else:
        print(f"The file '{args.file_path}/{args.file_name}.pkl' does not exist.")
        gt = {}

    # Disable Matplotlib's default 'q' for quit shortcut
    plt.rcParams['keymap.quit'].remove('q')

    # UI
    try:
        # iterate all people
        for person in os.listdir(args.data_folder):
            person_name = os.fsdecode(person)
            # if person_name not in gt:
            #     gt[person_name] = []

            dir_p = os.path.join(args.data_folder, person_name)
            for file in os.listdir(dir_p):
                filename = os.fsdecode(file)
                action_name = filename.split("_")[0]
                
                if filename.endswith(".csv") and filename not in labelled_files and person_name in args.p and action_name in args.a:
                    file_path = os.path.join(dir_p, filename)
                    print(file_path)
                    
                    # print("\nLaunching Peak Labeler. Close the window or press 'w' to exit.")
                    labeler = PeakLabeler(file_path=file_path, person=person, file_name=filename, window_size=30)
                    plt.show()

                    e, q, n = labeler.e, labeler.q, labeler.n

                    if e: # exit & not save
                        save = False
                        break
                    elif n: # skip current file
                        save = True
                        continue
                    else:
                        save = True
                        gt[filename]={'person': person_name,
                                      'action': action_name,
                                      'peaks_t': labeler.peaks,
                                      'peaks_i': labeler.indices,
                                      'file_len': labeler.file_len,
                                     }
                        print(f"Subject {person}, {filename} labelled.\n")

                        if q:
                            break
            if e or q:
                break
    except KeyboardInterrupt:
        save = False
        print("\nCtrl+C detected.")
    except Exception as e:
        # Handle any other general exception
        save = False
        print(f"An unexpected error occurred: {e}")
    finally:
        # This block always executes for cleanup
        if save:
            pickle.dump(
                gt,
                open(f'{args.file_path}/{args.file_name}.pkl', 'wb')
            )
            print(f"Saved, {args.file_path}/{args.file_name}.pkl")