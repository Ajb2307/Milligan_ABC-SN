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


# used:
# abcsn_config.SN_Stypes_int_to_str replaced with ABC_subtype_id_to_str
# to make the dictionary corresponding to the labels used above
# may just want to change the above to use the same strings as int_to_str function
ABC_subtype_id_to_str = {
    0: "Ia-norm",
    1: "Ia-91T",
    2: "Ia-91bg",
    3: "Iax",
    4: "Ib-norm",
    5: "Ibn",
    6: "IIb",
    7: "Ic-norm",
    8: "Ic-broad",
    9: "IIP",
}

subtype_to_ABC_ID= {
     'Ia-norm': 0,
     'Ia-91T': 1,
     'Ia-csm': None,
     'Ia-91bg': 2,
     'Ib-norm':4,
     'Iax':3,
     'Ia-pec': None,
     'Ic-norm':7,
     'IIP':9,
     'IIL': None,
     'IIb':6,
     'II-pec': None, ### ?
     'Ic-broad':8,
     'Ic-pec': None,
     'IIn': None,
     'Ibn': 5,
     'Ib-pec': None,
}


# takes in the SN subtype string given by milligan and returns the corresponding ABC ID
ABC_ID_dict ={"Ia": 0, # Ia-norm
          "Iap": 1, # Ia-91T ##### issue changed from 0 to 1 to match the ABC_subtype_id_to_str dictionary
          "Ic": 7, # Ic-norm
          "Ib": 4, # Ib-norm
          "II": 9, # IIP = type 2 plateau = normal
          "IIb": 6, # IIb
          "IIn": None, # no corresponding ABC ID 
          "SL": None,
          "TDE": None,
          "CRT": None,
          "CaRT": None
          }

# given the SN subtype string from milligan, returns the corresponding Milligan ID
# IDs used in Milligan to evaluate preformance 
Mill_ID_dict ={"Ia": 0, # Ia-norm
          "Iap": 0, # Ia-norm
          "Ic": 1, # Ib & Ic
          "Ib": 1, # Ib & Ic
          "II": 2, # IIP = type 2 plateau = normal
          "IIb": 1, # Ib & Ic
          "IIn": 2, # II
          "SL": 3, # SLSN
          "TDE": 4, # Non-SN
          "CRT": 4, # Non-SN
          "CaRT": 4  # Non-SN
          }

# five types recorded in Milligan et al.
Mill_types_to_int = {0: "Ia",
                     1: "Ib & Ic",
                     2: "II",
                     3: "SLSN",
                     4: "Non-SN",
                     5: "other"
                     }

# convert ABC types to Mill categories
ABC_to_Mill ={0:0,   # Ia-norm -> Ia
              1:0,   # Ia-91T -> Ia
              2:0,   # Ia-91bg -> Ia
              3:0,   # Iax -> Ia
              4:1,   # Ib-norm -> Ib & Ic
              5:1,   # Ibn -> Ib & Ic
              6:1,   # IIb -> Ib & Ic
              7:1,  # Ic-norm -> Ib & Ic
              8:1,  # Ic-broad -> Ib & Ic
              9:2,  # IIP -> II
            }
def get_Mill_ID_dict():   
    """Returns the dictionary mapping Milligan types to integers."""
    return Mill_types_to_int

def get_ABC_to_Mill_dict():
    """Returns the dictionary mapping ABC types to Milligan categories."""
    return ABC_to_Mill

def get_ABC_ID_dict():
    """Returns the dictionary mapping SN subtypes to ABC IDs."""
    return ABC_subtype_id_to_str

def get_ABC_Mill_type(SN_subtype_str):
    """Converts ABC type to Milligan category."""
    try:
      ABC_ID = ABC_ID_dict[SN_subtype_str]
      Mill_ID = Mill_ID_dict[SN_subtype_str]
      return ABC_ID, Mill_ID

    except Exception as e:
      print(SN_subtype_str)
      raise e