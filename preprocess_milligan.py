import sys

import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
from numpy.typing import NDArray

sys.path.insert(0, "/lustre/lrspec/users/4301/ABC-SN/code")
import data_degrading as dg
import abcsn_training
import abcsn_config
import data_preparation as dp
import preprocessing as pp


sys.path.insert(0, "/lustre/lrspec/users/4301/Milligan_project")
from plot_data import *
from process_data import *


from snid_sage.snid import preprocessing as snid_pp 

def folder_to_dataframe(folder, 
                        folder_loc = "/lustre/lrspec/users/4301/Milligan_ABC-SN/data/",
                        save_dir = "/lustre/lrspec/users/4301/Milligan_ABC-SN/data/folder_parquet",
                        num_wvl = 23411,
                        save = True,
                        save_suffix= "",
                        indicies = (None,None) ):
    
    filenames = sorted(glob.glob(folder_loc+ folder +"/*"))[indicies[0]: indicies[1]]
    file_info = [name.split("/")[-1] for name in filenames]
    df_metadata = pd.DataFrame(file_info, columns=["filename"])
    df_metadata[["host", "sn_type", "redshift", "SN_mag", "host_mag"]] = [get_filename_info(info) for info in file_info]
    df_metadata[ "host_mag"] = df_metadata[ "host_mag"].astype(float)
    df_metadata[ "redshift"] = df_metadata[ "redshift"].astype(float)
    df_metadata[ "SN_mag"] = df_metadata[ "SN_mag"].astype(float)

    # set up to extract the X values 
    num_wvl = 23411
    dat_size = len(df_metadata)
    plots = False
    
    X_all = np.zeros((dat_size, num_wvl))
    ABC_IDs = np.zeros((dat_size)) # classification as the float classifier
    Mill_IDs = np.zeros((dat_size)) # classification as the float classifier

    # extract flux and wavelength values from each file 
    for i, filename in enumerate(filenames):
        sep = " " # Define separator for reading data
        s = pd.read_csv(filename, sep=sep, header=None, comment="#") # Read the spectrum data
        wave = s[0]
        X_all[i] = s[1]

        # getting data 
        ABC_type, Mill_type = get_ABC_Mill_type(df_metadata["sn_type"].iloc[i])
        ABC_IDs[i] = ABC_type
        Mill_IDs[i] = Mill_type
        

    # combining metadata and spectra data into single df
    df_metadata["ABC_type"] = ABC_IDs
    df_metadata["Mill_type"] = Mill_IDs    
    df_metadata["Spectral Phase"] = np.zeros((dat_size))
    df_spec = pd.DataFrame(X_all, columns = wave)
    df_folder = pd.concat([df_metadata, df_spec], axis=1)
    df_folder = df_folder.drop(["filename"], axis = 1)

    if save:
        df_folder.columns = df_folder.columns.astype(str)
        df_folder.to_parquet(os.path.join(save_dir, folder+save_suffix+".parquet"))

    return df_folder

def interp_flux_logspace(log_wave, wvl0, fluxes0):
    # fluxes0 shape: (N_spectra, N_wavelengths)
    f = interp1d(wvl0, fluxes0, axis=1, bounds_error=False, fill_value=0.0)
    return f(log_wave)

def spectrum_logspace(wvl0, fluxes0, w0, w1):
    # need to change cadence and range of data points 
    R_current = np.mean(wvl0[:-1] / np.diff(wvl0)) # current resolution
    nw = dg.calc_num_wvl_bins(R_current, (2500, 10_000)) # number of wavelengths
    w0, w1 = 2_500, 10_000 # wavelength range
    log_wave, wvl0_bin_sizes = dg.calc_new_wvl(R_current, wvl0, (w0, w1)) 
    flux = interp_flux_logspace(log_wave, wvl0, fluxes0) # from logwave  
    return log_wave, flux 
    

def enforce_positive_flux(
    flux: NDArray[np.floating],
    *,
    eps_fraction: float = 1e-6,
) -> Tuple[NDArray[np.floating], float]:
    """
    based on snid_sage enforce_positive_flux from preprocessing with minor changes 
    
    Ensure that a flux array is strictly positive everywhere.

    This helper is designed for the SNID continuum fitter, which assumes
    non–negative real flux values when placing knots and taking log10
    averages.  It is intentionally unconditional: if *any* finite sample
    is ≤ 0, we shift the entire spectrum upward by a constant offset.

    Parameters
    ----------
    flux : array_like
        Input flux array (may contain negative/zero values and NaNs).
    eps_fraction : float, optional
        Small fractional safety margin relative to a robust flux scale
        (default 1e-6).  This avoids edge–case zeros after shifting.

    Returns
    -------
    shifted_flux : np.ndarray
        Flux array guaranteed to satisfy shifted_flux > 0 wherever the
        original data were finite.
    offset : float
        Constant added to the input flux (0.0 if no shift was required).
    """
    f = np.asarray(flux, dtype=float)
    if f.size == 0:
        return f.copy(), 0.0

    #find the start of the nonzero flux values
    flux_indices = np.where(~np.isclose(f, 0.0, atol=1e-16))[0]
    flux_start, flux_stop = flux_indices[0], flux_indices[-1]
    
    finite_mask = np.isfinite(f)
    if not np.any(finite_mask):
        return f.copy(), 0.0

    f_finite = f[finite_mask]

    # If fewer than 1% of finite samples are negative, treat this as a
    # normal fluxed spectrum and leave it unchanged. The continuum fitter
    # already ignores isolated negatives via its flux>0 masks.
    neg_frac = float(np.mean(f_finite < 0.0))
    if neg_frac < 0.01:
        return f.copy(), 0.0

    fmin = float(np.min(f_finite))
    if fmin > 0.0:
        # Already strictly positive; nothing to do.
        return f.copy(), 0.0

    # Robust scale for choosing a tiny positive epsilon.
    abs_vals = np.abs(f_finite)
    scale = float(np.median(abs_vals))
    if scale <= 0.0:
        # Fallback to max amplitude or 1.0 to avoid eps=0.
        scale = float(np.max(abs_vals)) if np.max(abs_vals) > 0.0 else 1.0

    # Guard against a *single* very deep negative outlier dominating fmin.
    # We clip the negative tail at -k * scale when determining the offset so that
    # continuum fitting still sees a strictly positive spectrum, but one or two
    # pathological pixels do not force an enormous upward shift that flattens all
    # real features (common in very long/high-resolution spectra).
    k = 1.2
    try:
        neg_clip_level = -k * scale
        f_finite_clipped = f_finite.copy()
        deep_neg_mask = f_finite_clipped < neg_clip_level
        if np.any(deep_neg_mask):
            _LOG.debug(
                "enforce_positive_flux: clipping %d deep negative sample(s) below %.3g "
                "to %.3g before computing offset",
                int(np.count_nonzero(deep_neg_mask)),
                float(f_finite_clipped[deep_neg_mask].min()),
                float(neg_clip_level),
            )
            f_finite_clipped[deep_neg_mask] = neg_clip_level
        fmin = float(np.min(f_finite_clipped))
    except Exception:
        # Fall back to the original minimum if clipping fails for any reason.
        fmin = float(np.min(f_finite))

    eps = float(eps_fraction) * scale
    if eps < 0.0:
        eps = 0.0

    offset = -fmin + eps

    # want to only shift nonzero data - ally
    shifted = f.copy()
    shifted[flux_start: flux_stop] += offset
    
    return shifted, float(offset)
    

def remove_continuum(flux, eps_fraction = 1e6):
    positive_flux, offset = enforce_positive_flux(flux, eps_fraction = 0.01)
    flattened_flux, cont = snid_pp.fit_continuum(positive_flux)
    return flattened_flux
remove_continuum_vec = np.vectorize(remove_continuum, signature='(n)->(n)')

def milligan_preprocessing(folder, # folder where spectra is located
                           file_indicies = (None, None), # processing all in the folder
                           plot_print = False, # plots the first flux at various stages 
                           snidify = True, # flattens spectra using a spline continuum 
                           save = True , # processed data is saved in a parquet 
                           save_suffix = "",
                           save_dir = "/lustre/lrspec/users/4301/Milligan_ABC-SN/data/folder_parquet", # directory where data is saved
                           phase_range=(-20, 50), # removed spectra outside this rage
                           ptp_range=(-np.inf, np.inf),
                           wvl_range=(4500, 7000), # wavelength where flux is examined
                            ): 

    df_folder = folder_to_dataframe(folder, indicies = file_indicies, save = save, save_suffix=save_suffix)
    if plot_print:
        print("data loaded into dataframe")
    df_prep = preproccess_dataframe(df_folder, phase_range, ptp_range, wvl_range)
    # if save:
        # decided this is not a necessary step to save
        #df_prep.to_parquet(os.path.join(save_dir, folder + save_suffix + "_preprocessed.parquet"))
    if plot_print:
        print("data preprocessed")
        df_folder.loc[0].iloc[8:].T.plot()
        plt.title("orginal data preprocessed")
        plt.show()

    # extracting data
    R = 100
    data = dp.extract_dataframe(df_prep)
    index = data[0]  # SN Name for each spectrum
    wvl0 = data[1]  # Wavelength array
    flux0_columns = data[2]  # Columns that index the fluxes in the dataframe
    df_metadata = data[5]  # Sub-dataframe containing only the metadata
    fluxes0 = data[6]  # Only the flux values in a numpy array

    # rebinning the spectra to be in a logspace 
    w0, w1 = 2_500, 10_000 # wavelength range
    log_wave, flux =  spectrum_logspace(wvl0, fluxes0, w0, w1)

    if plot_print:
        print("Data put into logspace")
        plt.plot(log_wave, flux[0], label = "logspace data")
        plt.plot(wvl0, fluxes0[0], alpha =0.5, label = "preprocessed data")
        plt.legend()
        plt.show()

    # normalize spectra using snid 
    if snidify:
        # normalize the spectra 
        flux = remove_continuum_vec(flux)

        if plot_print:
            plt.plot(log_wave, flux[0], label = "flattened data")
            plt.legend()
            plt.show()
            
    
    ## code from data_degrade.degrade_dataframe

    # Perform degradation (i.e. lowers resolution) for each spectrum in the dataset. The function
    # # degraded_spectrum is vectorized, so supplying multiple spectrum allows
    # # the operation to be parallelized.
    fluxes_convolve, wvl_degraded, fluxes_degraded = dg.degrade_spectrum(R, log_wave, flux)
    
    # `wvl_degraded` is actually multiple copies of the same wavelength array, 
    # so we only need the first one.
    degraded_flux_columns = wvl_degraded[0].astype(str)
    
    # Store the convolution data (not the rebinned spectra)
    sn_data_convolve = pd.DataFrame(data=fluxes_convolve,
                                      columns=log_wave,
                                      index=index,
                                      dtype=float)
    sn_data_convolve = pd.concat([df_metadata, sn_data_convolve], axis=1)
    sn_data_convolve.columns = sn_data_convolve.columns.astype(str)
    
    # Store the rebinned spectra into its own dataframe.
    sn_data_degraded = pd.DataFrame(data=fluxes_degraded,
                                      columns=degraded_flux_columns,
                                      index=index,
                                      dtype=float)
    sn_data_degraded = pd.concat([df_metadata, sn_data_degraded], axis=1)
    sn_data_degraded.columns = sn_data_degraded.columns.astype(str)
    
    if save:
        sn_data_convolve.to_parquet(os.path.join(save_dir, folder + save_suffix + "_convoloved.parquet"))
        sn_data_degraded.to_parquet(os.path.join(save_dir, folder + save_suffix + "_rebinned.parquet"))
    
    if plot_print:
        print("data has been degraded to R = 100")
        #shifted data to new wavelength axis but not degraded 
        plt.plot(log_wave, fluxes_convolve[0], label = "convolved fluxed")
        plt.plot(wvl_degraded[0], fluxes_degraded[0], label = "degraded fluxed")

