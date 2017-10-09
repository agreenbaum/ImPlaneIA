#! /usr/bin/env python
from nrm_analysis import InstrumentData, nrm_core

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'../../'))
print(sys.path)


# define data files
targfiles = [f for f in os.listdir("../f430_data") if "tcube" in f] # simulated data
calfiles = [f for f in os.listdir("../f430_data") if "ccube" in f] # simulated data

# set up instrument-specfic part
nirissdata = InstrumentData.NIRISS(filt="F430M", objname="targ")


ff =  nrm_core.FringeFitter(nirissdata, oversample = 3, savedir="targ", datadir="../f430_data", 
                            npix=35, interactive=False)
ff.fit_fringes(targfiles)
    
ff2 =  nrm_core.FringeFitter(nirissdata, oversample = 3, savedir="cal1", datadir="../f430_data", 
                            npix=35, interactive=False)
ff2.fit_fringes(calfiles)


