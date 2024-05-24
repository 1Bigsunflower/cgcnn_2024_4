#!/bin/bash

original_dir=$(pwd)

init_csv="cgcnn_init.csv"
emb_csv="cgcnn_emb.csv"

rm "$init_csv"
rm "$emb_csv"

dims=(1 2 4 8 16 32 64 128)

touch "$init_csv"

for dim in "${dims[@]}"; do
    log_file="../cgcnn_lightning/cgcnn_dim${dim}.log"
    grep "test_MAE" "$log_file" | awk '{printf "%s", $4; if(NR%5==0) printf "\n"; else printf ","} END {if(NR%5!=0) printf "\n"}' >> "$init_csv"
    cd "$original_dir" || exit 1
done


touch "$emb_csv"
for dim in "${dims[@]}"; do
    log_file="../cgcnn_lightning_emb/cgcnn_dim${dim}_emb.log"
    grep "test_MAE" "$log_file" | awk '{printf "%s", $4; if(NR%5==0) printf "\n"; else printf ","} END {if(NR%5!=0) printf "\n"}' >> "$emb_csv"
    cd "$original_dir" || exit 1
done