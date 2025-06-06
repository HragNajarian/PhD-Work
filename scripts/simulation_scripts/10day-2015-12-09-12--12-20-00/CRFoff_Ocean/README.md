These tests within this folder will utilize the same WPS configuration as the CTRL

Simply put, icloud is set to 0 to turn off cloud-radiative interactions...
Additional modificiations to module_ra_rrtmg_lw.F and ...sw.F files turn off cloud-radiative interactions
	over ocean only.

For simplicity, everything is the EXACT same as NCRF, but with edits done on the module_ra_rrtmg_lw.F and ...sw.F files.

namelists.input are specific to when in time the interaction is turned off

These tests start from restart files which means the namelist.input also is edited to accomidate this