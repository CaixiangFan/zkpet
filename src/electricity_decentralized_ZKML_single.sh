#!/bin/bash

# Define the base folder where the folders are located
base_folder="/mnt/zkpet/checkpoints"

# fix site 40
site=40
model='NLinear'
# Loop through historical length
# for i in 96 192 336 720
for i in 96 192 336 720
do
    folder_name="$base_folder/Electricity_${i}_24_${model}_custom_ftS_tg${site}_sl${i}_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/"
    # Check if the folder exists
    if [ -d "$folder_name" ]; then
        # Change directory to the folder
        cd "$folder_name"
        echo "Changed to $folder_name"
        
        # Create the setup
        ezkl gen-settings -M network.onnx
        ezkl calibrate-settings -M network.onnx -D input.json --target resources
        ezkl get-srs --srs-path=kzg.srs -S settings.json
        ezkl compile-circuit -M network.onnx -S settings.json --compiled-circuit network.ezkl
        ezkl setup -M network.ezkl --srs-path=kzg.srs

        # Prove
        ezkl gen-witness -D input.json -M network.ezkl
        /usr/bin/time -v -o perf.txt ezkl prove --witness witness.json -M network.ezkl --proof-path model.pf --pk-path pk.key --srs-path=kzg.srs
        # break  # Exit the loop after finding the first matching folder
    fi
done >> log.txt