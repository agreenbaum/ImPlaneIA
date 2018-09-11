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
