import os as _os

def write_condor_files(home_folder,uname='simone.mastrogiovanni',
agroup='ligo.dev.o4.cbc.hubble.icarogw',memory=10000,cpus=1,disk=10000):
    '''
    This function looks for all the *.py files in a folder and write a set of condor files
    needed for submission on write_condor_files. To launch the jobs, 1) Generate files with this function
    2) run chmod +x *.sh 3) launch the sub files.

    Parameters
    ----------
    home_folder: str
        Folder where to look for python files
    uname: str
        Username for condor
    agroup: str
        Accounting group for condor
    '''
    list_py_files = _os.listdir(home_folder)

    for file in list_py_files:
        if file.endswith('.py'):
            if file=='config.py':
                continue
            fname = file[:-3:1]

            f = open(home_folder+fname+'.sh', 'w')
            f.write('#!/bin/bash')
            f.write('\n')
            f.write('MYJOB_DIR='+home_folder)
            f.write('\n')
            f.write('cd ${MYJOB_DIR}')
            f.write('\n')
            f.write('python '+file)
            f.close()

            f = open(home_folder+fname+'.sub', 'w')
            f.write('universe = vanilla\n')
            f.write('getenv = True\n')
            f.write('executable = '+home_folder+fname+'.sh\n')
            f.write('accounting_group = '+agroup+'\n')
            f.write('accounting_group_user = '+uname)
            f.write('\n')
            f.write('request_memory ='+str(memory)+'\n')
            f.write('request_cpus ='+str(cpus)+'\n')
            f.write('request_disk ='+str(disk)+'\n')    
            f.write('output = '+home_folder+fname+'.stdout\n')
            f.write('error = '+home_folder+fname+'.stderr\n')
            f.write('queue\n')
            f.close()
            _os.system('chmod a+x '+home_folder+'*.sh')
