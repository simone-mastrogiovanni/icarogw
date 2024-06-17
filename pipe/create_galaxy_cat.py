import icarogw
icarogw.utils.write_all_scripts_catalog(home_folder='/home/simone.mastrogiovanni/GLADEp_tests',
                                       outfolder='/home/simone.mastrogiovanni/GLADEp_tests/pixelated_gladep',
                                       nside=64,
                                       fields_to_take=['ra','dec','m_K','z','sigmaz'],
                                       grouping='K-band',
                                       apparent_magnitude_flag='m_K',
                                       nside_mthr=32,
                                       mthr_percentile=50.,
                                       Nintegration=np.logspace(-3,0.5,4000),
                                       Numsigma=3,
                                       zcut=0.5,
                                       outfile='/home/simone.mastrogiovanni/GLADEp_tests/icaro_gladep.hdf5',
                                       subgrouping='eps-1',band='K', epsilon=1., NumJobs=20)