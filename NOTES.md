
BETA- ACHIEVED FOR anand0xff affine

AZG-AS skype 2018.08.24:

Move to beta (by Sep 7) involves:

	Alex to make afine_dev branch on github/agreenbaum
	
	Anand to pull affine_dev to local
	
	Anand to unify affine_dev & affine into affine_dev
	
	Anand to run first five boxes notebooks/NIRISS_AMI_tutorial.ipynb

2018.09.11 regrouping:

	who is going to integrate my current code with your post-OIfits-creation code?  
	I might suggest deleting the affine_dev branch, re-cloning it. and replacing your
	fringe determining files with mine.  
	
	What would be good is to have some short quick-to-run test code to check that
	things still work downstream of OIfits creation. 

	There are 'small' technical things to finish also - I've only done rotation-finding,
	and still need to implement finding something else (pixel scale?).
	Find the change from nominal pixel scale.

	So direction from you would help me. 

	One not-so-small thing to do is to write test code to check all the psf_offset
	implementations (I see lines like ImCtr = np.array((psf_offset[1], psf_offset[0]))
	here and there.  It should be unified across the calcPSF and make_model sections,
	and documented clearly.  This is probably best done when we look at the code
	simultaneously, skyping at the same time.


Note: Anand to just do JWST NIRISS integration, 
(Alex to follow up on other instruments later)

New_and_better_scaling and auto_find_center [to be] removed 

AMI paper writing focus ~mid September?

Deepashri involved in test exercising/writing whenever (month or two?)

Dec 13 2018 AZG AS skype.  Next meeting ~1 week
  
	Calib initializing - try abs path names
	Cut down targ & cal to a few (eg 3) slices each for faster  debugging
	Examine residual fits w/screen share w/AZG
	
Dec 14 2018 AS

	Fixed path fragility of datadir - no defaulting to "."  (could have repercussions for other instruments?)
	Created two-slice cubes in example_data/niriss_niriss
	Adjusted file names and datadir value in ami notebook
	Notebook now runs through to ascii fringe info output, 
	residuals look ~antisymmetric (flip and invert a second residuals*.fits in ds9, then blink)
	
	Notebook fails in 
	In [8]  calib.save_to_oifits("exampleoifitsfiles.oifits") 
	~/gitsrc/agreenbaum/ImPlaneIA/nrm_analysis/misctools/write_oifits.py in oi_data(self, read_from_txt, **kwargs)
    332         #print "cps given to oifits writer, again:"
    333         #print self.t3phi
	--> 334         self.t3flag[abs(self.t3phi)>self.phaseceil]=1
    335         for i in range(int(self.ncps)):
    336             """self, timeobs, int_time, t3amp, t3amperr, t3phi, t3phierr, flag, u1coord,

	IndexError: boolean index did not match indexed array along dimension 0; dimension is 1 but corresponding boolean dimension is 2

	
