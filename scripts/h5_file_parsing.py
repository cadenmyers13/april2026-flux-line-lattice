"""
h5_file_parsing.py
This script provides utilities for parsing and visualizing data from HDF5 files. 
It includes functions to extract resistance and temperature data along with their 
corresponding time values, and to plot these data sets using matplotlib.
The script is designed to work with HDF5 files that follow a specific structure, 
where resistance and temperature data are stored under predefined paths. Users can 
run the script from the command line, providing the path to an HDF5 file as an argument, 
to generate plots for resistance vs. time and temperature vs. time.
Functions:
-----------
- open_hdf5(file_path): Opens an HDF5 file and returns the file object.
- get_resistance_and_time(file_path): Extracts resistance and time data from the HDF5 file.
- get_vti_temp_vs_time(file_path): Extracts temperature and time data from the HDF5 file.
- plot_resistance_vs_time(file_path): Plots resistance vs. time using matplotlib.
- plot_vti_temp_vs_time(file_path): Plots temperature vs. time using matplotlib.
Usage:
------
Run the script from the command line with the path to an HDF5 file as an argument:
    python h5_file_parsing.py /path/to/hdf5_file.h5

"""

import h5py
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def open_hdf5(file_path):
    """Open an HDF5 file and return the file object.
    
    Parameters
    ----------
    file_path : Path
        The path to the HDF5 file.
    
    Returns
    -------
    h5py.File
        The opened HDF5 file object."""
    return h5py.File(file_path, 'r')

def get_resistance_and_time(file_path):
    """Extract resistance and time data from the HDF5 file.
    
    Parameters
    ----------
    file_path : Path
        The path to the HDF5 file.
    
    Returns
    -------
    resistance : np.ndarray
        The array of resistance values.
    time : np.ndarray
        The array of time values corresponding to the resistance measurements.
    """
    with open_hdf5(file_path) as f:
        resistance = f["entry"]['DASlogs']['CryoSensorC']['value'][:]
        time = f["entry"]['DASlogs']['CryoSensorC']['time'][:]
    return resistance, time



def get_vti_temp_vs_time(file_path):
    """Extract temperature and time data from the HDF5 file.
    
    Parameters
    ----------
    file_path : Path
        The path to the HDF5 file.
    
    Returns
    -------
    temperature : np.ndarray
        The array of temperature values.
    time : np.ndarray
        The array of time values corresponding to the temperature measurements.
    """
    with open_hdf5(file_path) as f:
        base = f["entry"]['DASlogs']['CG2:SE:CryoG:TempActual']
        temperature = base['value'][:]
        time = base['time'][:]
    return temperature, time

def plot_resistance_vs_time(file_path):
    """Plot resistance vs. time using matplotlib.
    
    Parameters
    ----------
    resistance : np.ndarray
        The array of resistance values.
    time : np.ndarray
        The array of time values corresponding to the resistance measurements.
    """
    resistance, time = get_resistance_and_time(file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(time, resistance, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Resistance (Ohms)')
    plt.title(f"{file_path.name}")
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_vti_temp_vs_time(file_path):
    """Plot temperature vs. time using matplotlib.
    
    Parameters
    ----------
    temperature : np.ndarray
        The array of temperature values.
    time : np.ndarray
        The array of time values corresponding to the temperature measurements.
    """
    temperature, time = get_vti_temp_vs_time(file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(time, temperature, marker='o', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature (K)')
    plt.title(f"{file_path.name}")
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Open and explore an HDF5 file.")
    parser.add_argument("file", help="Path to the HDF5 file to open")
    args = parser.parse_args()
    file_path = Path(args.file)

    plot_resistance_vs_time(file_path)
    plot_vti_temp_vs_time(file_path)



if __name__ == "__main__":
    main()