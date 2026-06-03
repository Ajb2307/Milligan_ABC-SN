import sys

import numpy as np
import keras
import pandas as pd
import matplotlib.pyplot as plt


sys.path.insert(0, "/lustre/lrspec/users/4301/ABC-SN/code")
from data_degrading import degrade_spectrum
import abcsn_training
import abcsn_config
from preprocessing import preproccess_dataframe

sys.path.insert(0, "/lustre/lrspec/users/4301/snidpy/sourcepy")
from apodize import *
from logwave import Logwave as lw
from logwave import log_rebin

lw.mean_zero = meanzero # wonder if this will fix annoying print

host_types = ["elliptical", "s0", "sa", "sb", "sc", "starb1", "starb1", "starb2", "starb3", "starb4", "starb5", "starb6"]
# needs to go longest to shortest so it doesnt find II instead of IIb
# there's Iap?? not mentioned in paper

sn_class_dict = {"IIb": "IIb", 
                 "IIn": "IIn", 
                 "II_": "II", 
                 "Iap": "Iap", 
                 "Ia_": "Ia", 
                 "Ic_": "Ic", 
                 "Ib_": "Ib", 
                 "SL_": "SL", 
                 "TDE": "TDE", 
                 "CRT": "CaRT"}

                 
def get_host_sn_type(fileinfo):
  """Extracts the host morphology and true SN class from the filename.
  filename format: [Host Morphology][True SN Class]Smag...
  """
  host = None
  sn = None
  info = fileinfo.split("Smag")[0]
  host = info[:-3]
  assert host in host_types, "Host type not in list: {}".format(fileinfo[0])

  sn = info[-3:]
  assert sn in sn_class_dict.keys(), "SN type not in list: {}".format(fileinfo[0])
  return host, sn_class_dict[sn]

def get_redshift(fileinfo):
  """Extracts the redshift from the filename.
  filename format: ...z[Redshift]texp...
  """
  redshift = fileinfo.split("z")[1].split("texp")[0]
  redshift = float(redshift)
  assert redshift > 0, "Redshift must be positive: {}".format(fileinfo[0])
  assert redshift < 3, "Redshift must be less than 3: {}".format(fileinfo[0]) # based on this dataset specifically
  return redshift

def get_SN_mag(fileinfo):
  """Extracts the SN Fibre Magnitude from the filename.
  filename format: ...Smag[SN Fibre Mag.]Gmag...
  """
  Smag = fileinfo.split("Smag")[1].split("Gmag")[0]
  Smag = float(Smag)
  return Smag

def get_host_mag(fileinfo):
  """Extracts the Host Fibre Magnitude from the filename.
  filename format: ...Gmag[Host Fibre Mag.]z...
  """
  Gmag = fileinfo.split("Gmag")[1].split("z")[0]
  Gmag = float(Gmag)
  return Gmag

def get_filename_info(filename):
  """Extracts all the information from the filename.
  filename format: [Host Morphology][True SN Class]Smag[SN Fibre Mag.]Gmag[Host Fibre Mag.]z[Redshift]texp[Exposure Time in Mins].txt
  """
  host, sn_type = get_host_sn_type(filename)
  redshift = get_redshift(filename)
  Smag = get_SN_mag(filename)
  Gmag = get_host_mag(filename)
  return host, sn_type, redshift, Smag, Gmag


def simplified_process_spectrum(wave, flux): #-> Dict:
    """Process all epochs for a supernova.
      from snidpy repo process_spectrum but simplified to take data not file"""

    from logwave import SNIDConfig

    # Initialize arrays
    nw = SNIDConfig().nw
    flog = np.zeros(nw)
    fnorm = np.zeros(nw)
    fmean = [0]
    nknot = np.zeros(1, dtype=int)
    
    # Setup log wavelength grid
    dwlog = np.log(SNIDConfig().w1 / SNIDConfig().w0) / nw
    wlog = SNIDConfig().w0 * np.exp(np.arange(nw + 1) * dwlog)
    
    # Process each epoch
    most_knots = 0
    # Read spectrum
    
    
    assert len(wave) > 0, "no data"
        
    
    # Rebin to log scale
    flog = log_rebin(wave, flux, nw, SNIDConfig().w0, SNIDConfig().w1)
    
    # Normalize (remove continuum)
    fnorm, l1, l2, nknot, xknot, yknot = meanzero(flog)
    #print("here", fnorm)
    most_knots = max(most_knots, nknot)
    
    # Apodize
    #fnorm = apodize(fnorm, 10,
    #                      -10, self.config.percent)
    
    # Calculate mean flux for scaling
    fmean = np.mean(flog)
    # flog = flog / fmean
    
    # Prepare output
    output = {
        'wlog': wlog,
        'flog': flog,
        'fnorm': fnorm,
        'fmean': fmean,
        'nknot': nknot
    }
    return output

# used to process milligan files 
def process_files(filename, wmin, wmax, plot_spectra = True, verbose = True,  snidify = True, R =100):

  try:
    sep = " " # Define separator for reading data
    s = pd.read_csv(filename, sep=sep, header=None, comment="#") # Read the spectrum data
    if plot_spectra:
      display(s.head())
    
    # Read and plot the original spectrum
    wave, flux = s[0], s[1]
    # Apply wavelength mask
    mask = (wave >= wmin) & (wave <= wmax)
    wave, flux = wave[mask], flux[mask]
    sp = wave, flux
    
    if plot_spectra:
      name = filename.split("/")[-1]
      s.plot(x=0, y=1)
      plt.title(name)
      plt.plot(sp[0], sp[1], 'k-.', alpha=0.5)
      plt.show()
    if verbose:
      print("READ IN OK")


    if snidify == True: # removes the continuum using 13(?) degree polynomial 
      # Process the spectrum using logwave (lw) module functions
      processedsn = simplified_process_spectrum(wave, flux)
      if verbose:
        print("processed")
      # print(processedsn)
      
      # Extract and normalize flux values
      # this is adjusting the wavelength bins to take the center value
      wlog0 = processedsn["wlog"][0:-1] + np.diff(processedsn["wlog"]) / 2
      flog = processedsn["fnorm"]


    if snidify == False: # does not remove continuum
      processedsn = simplified_process_spectrum(wave, flux)
      wlog0 = processedsn["wlog"][0:-1] + np.diff(processedsn["wlog"]) / 2
      flog = processedsn["flog"]

    # select correct range
    select = (wlog0 < wmin) | (wlog0 > wmax)

    ## this doesnt seem to do anything
    # normalize log flux
    flog = (flog - flog[~select].mean()) / flog[~select].std()
    flog[select] = 0

    # Degrade the spectrum
    spd = degrade_spectrum(R,
        wlog0,
        flog
        )

    # Plot the processed and degraded spectra
    if plot_spectra:
      plt.title(name)
      plt.plot(wlog0, flog)
      plt.plot(spd[1], spd[2])

    # return processed spectrum data and extract metadata
    X = np.array([spd[2]])
    X = X.reshape(1,X.shape[1])

    wvl = spd[1]

    return wvl, X
 
  except Exception as e:
    print(filename, "FAILED")
    print(e)


# used to try and reprocess abcsn data
def process_data(X, wvl, wmin, wmax, plot_spectra = True, verbose = True, use_preprocess = True,snidify = True, R =100):
  

    wave, flux = wvl, X
    # Apply wavelength mask
    mask = (wave >= wmin) & (wave <= wmax)
    wave, flux = wave[mask], flux[mask]
    sp = wave, flux
    
    if plot_spectra:
      plt.plot(sp[0], sp[1], 'k-.', alpha=0.5)
      plt.show()
    if verbose:
      print("READ IN OK")

    if snidify == True:
      # Process the spectrum using logwave (lw) module functions
      processedsn = simplified_process_spectrum(wave, flux)
      if verbose:
        print("processed")
      # print(processedsn)
      
      # Extract and normalize flux values
      # this is adjusting the wavelength bins to take the center value
      wlog0 = processedsn["wlog"][0:-1] + np.diff(processedsn["wlog"]) / 2
      flog = processedsn["fnorm"]
      # print(flog.mean())
    
    if snidify == False:
      # Process the spectrum using logwave (lw) module functions
      processedsn = simplified_process_spectrum(wave, flux)
      wlog0 = processedsn["wlog"][0:-1] + np.diff(processedsn["wlog"]) / 2
      flog = processedsn["flog"]
      # print(flog.mean())

    # select correct range
    select = (wlog0 < wmin) | (wlog0 > wmax)

    # normalize log flux
    flog = (flog - flog[~select].mean()) / flog[~select].std()
    flog[select] = 0

    # Degrade the spectrum
    spd = degrade_spectrum(R,
        wlog0,
        flog
        )

    # Plot the processed and degraded spectra
    if plot_spectra:
      # plt.title(name)
      plt.plot(wlog0, flog)
      plt.plot(spd[1], spd[2])

    # return processed spectrum data and extract metadata
    X = np.array([spd[2]])
    X = X.reshape(1,X.shape[1])

    wvl = spd[1]

    return wvl, X

  