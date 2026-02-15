#!/bin/bash



conda activate rdkit

#preparing mol2s - must be run by Mohsen: Be careful here and only run this for new molecules by adjusting the code for the specific folder. Otherwise you need to double do every forlder since a randomly chosen "forcefield" folder is going to be copied in all the folders. 
#cd /afs/crc/group/whitmer/Data-MF-04/amber
#bash run_antechamber.sh

# mv mol2s to "/afs/crc/group/whitmer/Data-MF-05/ML/traj" directory s - must be run by Mohsen
rm *.mol2
cd /afs/crc/group/whitmer/Data-MF-05/ML/traj
bash mv_mol2.sh

# remove duplicated ones of the default mol2 that was in "/afs/crc/group/whitmer/Data-MF-05/amber/forcefield" directory, must be run by Mohsen
bash delete-mol2.sh

# mv mol2s and other inputs to their corresponding folders in "/afs/crc.nd.edu/user/m/mfarshad/Private/ML/traj" directorys - must be run by Mohsen
cd /afs/crc.nd.edu/user/m/mfarshad/Private/ML-new/traj
rm -r host-*
bash mv_files.sh

# empty all-host-props folder
rm all-host-props/*

# empty all-host-props-nopore folder
rm all-host-props-nopore/*

# empty all-host-mol2s folder from mol2 files
rm all-host-mol2s/*.mol2

# transfer mol2 files into mol2 folder
for i in $(ls host-*/host_*.mol2); do
    cp $i all-host-mol2s/
done

# add resname to mol2 file: Note that this is step is not necessary anymore, since the updated mol2 files already have resnames
#for x in host-*/*.mol2; do python add-res-name-to-mol2.py $x; done

# fix mol2 files - change atom names from GAFF atom types to atom names
cd all-host-mol2s/
for x in host_*.mol2; do (./change_names_mol2.sh $x "fixed_$x"); done
cd ..

# get rdkit peoperties from mol2 files
for x in all-host-mol2s/fixed_host_*.mol2; do python get-host-prop-v09.py $x; done

# # copy trajectories from host folders to pore_diameter - must be run by Mohsen
# 'for x in host-*; do (cp $x/*trr $x/*gro $x/*mol2 pore_diameter/) done
# cp pore_diameter.py pore_diameter/'

# # get pore properties and add them to the host properties - must be run by Mohsen
# # cd pore_diameter, be careful this is a high load job that can create stress on the CRC which results in a headsup by them.
# 'bash run_python.sh'

# # transfer pore diameter files into mol2 folder
# '
# for i in $(ls host-*/*.csv); do
#     cp $i pore_diameter/
# done
# '

# run pore diameter calculations one at a time
#for x in host-*; do (cd $x; cp ../pore_diameter_single.py .; python pore_diameter_single.py); done
for x in host-*; do (cd $x; python pore_diameter_single.py); done

# submit as job all at once from Private/ML/traj directory
#./run_python.sh

# for x in */diameter_profile*; do python combine-all-props.py $x; done # this version adds avg and std pore diameter
for x in host-*/diameter_profile*; do python combine-all-props-v02.py $x; done # this version adds only avg pore diameter

# get the free energy values
cd targets
./run_wham.sh
./run_python.sh # must be run by Mohsen
cd ..

# combine host properties without targets
python combine-props-targets-v05.py

# combine host properties with targets
python combine-props-targets-v06.py

# to create paper data points of all host (H1-H36)
cd all-host-props-nopore
./delete_difference.sh
cd ..

# pca extraction
cd pca 
python pca-v03.py # unlabeled
python pca-v04.py # labeled
python pca-paper_fig.py # for figure 6 in paper
cd ..

# GP
python gpr-random_seed-paper.py # for figure 7 in paper
python gpr-pca-active-learning-v03.py # creates data for tables 2 and S4
