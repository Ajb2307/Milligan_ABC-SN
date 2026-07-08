import sys
import argparse
sys.path.insert(0, "/lustre/lrspec/users/4301/Milligan_project")
from preprocess_milligan import *
import glob
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=str, required=True, help='Sample folder name')
    args = parser.parse_args()
    
    folder = args.sample
    folder_loc = "/lustre/lrspec/users/4301/Milligan_ABC-SN/data/"  
    
    # Check if folder exists
    full_path = os.path.join(folder_loc, folder)
    if not os.path.exists(full_path):
        print(f"Error: {full_path} does not exist")
        sys.exit(1)
    
    file_num = len(glob.glob(full_path + "/*"))
    print(f"Processing {folder}: {file_num} files found")
    
    indicie_division = [i for i in range(0, file_num, 100)]
    if file_num > indicie_division[-1]:
        indicie_division.append(file_num)
    
    for i in range(len(indicie_division) - 1):
        start = indicie_division[i]
        end = indicie_division[i+1] 
        
        milligan_preprocessing(folder, # folder where spectra is located
                                file_indicies = (start, end), 
                                plot_print = False,
                                save_suffix = f"_{i}", 
                                save = True)

        print(f"Processing {folder} with indices {start} to {end}")

if __name__ == "__main__":
    main()