#!/bin/bash

# Define the base folder where the folders are located
base_folder="/mnt/zkpet/checkpoints"

# Loop through numbers from 0 to 319
for i in 0 2 6 81 104 114 115 116 117 118 119 121 122 125 146 182 298
do
    folder_name="$base_folder/Electricity_192_24_Linear_custom_ftS_tg${i}_sl192_ll48_pl24_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/"
    # Check if the folder exists
    if [ -d "$folder_name" ]; then
        # Change directory to the folder
        cd "$folder_name"
        echo "Changed to $folder_name"
        
        # Create the setup
        ezkl gen-settings -M network.onnx
        ezkl get-srs --logrows 23 --srs-path=23.srs
        ezkl calibrate-settings -M network.onnx -D input.json --target resources
        ezkl compile-circuit -M network.onnx -S settings.json --compiled-circuit network.ezkl
        ezkl setup  -M network.ezkl --srs-path=23.srs --vk-path=vk.key --pk-path=pk.key 

        # Prove
        ezkl gen-witness -D input.json -M network.ezkl
        /usr/bin/time -v -o perf.txt ezkl prove --proof-type=for-aggr -W witness.json -M network.ezkl --proof-path site${i}.pf --srs-path=23.srs  --pk-path=pk.key
        # break  # Exit the loop after finding the first matching folder
    fi
done >> log.txt