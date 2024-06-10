## Follow these steps to install ICAROGW of a specific branch on MacOs

### Step 1:
conda create -n env_name python=3.10

### Step 2: 
conda activate env_name

### Step 3: 
mdkir dossier_icarogw  
cd dossier_icarogw

### Step 4:
git clone git@github.com:icarogw-developers/icarogw.git  
cd icarogw

### Step 5: (To switch to an other branch)
git checkout branch_name 

### Step 6:
pip install . --force

### Step 7: (Optionnal, only if hdf5 error while step 6)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

Add follow the instructions to add brew to the Path

### Step 8: 
brew reinstall hdf5  
export export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.14.3/

### Step 9:
pip install . --force
