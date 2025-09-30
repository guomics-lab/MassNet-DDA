species_list=(
    "test_DDA_18species_Acinetobacter_calcoaceticus"
    "test_DDA_18species_Aedes_aegypti"
    "test_DDA_18species_Arabidopsis_thaliana"
    "test_DDA_18species_Aspergillus_fumigatus"
    "test_DDA_18species_Bacteroides_thetaiotaomicron"
    "test_DDA_18species_Caenorhabditis_elegans"
    "test_DDA_18species_Drosophila_melanogaster"
    "test_DDA_18species_Escherichia_coli"
    "test_DDA_18species_Helicobacter_pylori"
    "test_DDA_18species_Homo_sapiens"
    "test_DDA_18species_Marchantia_polymorpha"
    "test_DDA_18species_Mus_musculus"
    "test_DDA_18species_Pyrococcus_furiosus"
    "test_DDA_18species_Saccharomyces_cerevisiae"
    "test_DDA_18species_Toxoplasma_gondii"
    "test_DDA_18species_Trypanosoma_brucei"
    "test_DDA_18species_Zea_mays"
    "test_DDA_18species_Solanum_lycopersicum"
)
XuanjiNovo_100M=/mnt/shared-storage-user/beam/zhangxiang/pptm/xusheng/MasNet_100M_Cont/epoch=695-step=324000.ckpt
GlanceNovo_30M=/mnt/shared-storage-user/beam/zhangxiang/pptm/xusheng/NON944/epoch=23-step=20000.ckpt 
test_path_l=/mnt/shared-storage-user/beam/zhangxiang/dump3/test_325d27b5225f45838d7cf1ffb4559794
bacillus=/mnt/shared-storage-user/beam/denovo/Scaling/XIangDia2/MassNet-DDA/bacillus.10k.mgf
# for species in "${species_list[@]}"; do
#     export SPECIES_NAME="$species"
#     full_path="$base_dir/$species"
#     echo "Running evaluation on $full_path"
#     torchrun --nproc_per_node=8  -m XuanjiNovo.XuanjiNovo --config=./inf3.yaml \
#         --mode=eval \
#         --model=$XuanjiNovo_100M \
#         --peak_path=$test_path_l
# done

MASTER_PORT=${MASTER_PORT:-$(( ( RANDOM % 10000 ) + 20000 ))}
echo "Using MASTER_PORT=$MASTER_PORT"

torchrun --nproc_per_node=4 --master_port=${MASTER_PORT} -m XuanjiNovo.XuanjiNovo  \
        --mode=denovo \
        --model=$XuanjiNovo_100M \
        --peak_path=$bacillus --gpu=0,1,2,3