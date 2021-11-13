# Attacks
declare -a attacks=(
  # 'CW'
  # 'PGD'
  # 'ODI'
  # 'MD'
  # 'MDMT'
  # 'AA'
  # 'MDE'
  # "MDMT+"
  # "DLR"
  # "MT"
  # 'DLRMT'
  'MD'
  'MD_first_stage_step_search_05'
  'MD_first_stage_step_search_10'
  'MD_first_stage_step_search_15'
  'MD_first_stage_step_search_20'
  'MD_first_stage_step_search_25'
  'MD_first_stage_step_search_30'
  'MD_first_stage_step_search_35'
  'MD_first_stage_step_search_40'
  'MD_first_stage_step_search_45'
  'MD_first_stage_step_search_50'
  'MD_first_stage_initial_step_size_search_02'
  'MD_first_stage_initial_step_size_search_04'
  'MD_first_stage_initial_step_size_search_06'
  'MD_first_stage_initial_step_size_search_08'
  'MD_first_stage_initial_step_size_search_10'
  'MD_first_stage_initial_step_size_search_12'
  'MD_first_stage_initial_step_size_search_14'
  'MD_first_stage_initial_step_size_search_16'
  'MD_first_stage_initial_step_size_search_18'
  'MD_first_stage_initial_step_size_search_20'
)

# Submmit All
declare -a defence=(
  # "RST"
  # "TRADES"
  # "MART"
  # "MMA"
  "BAT"
  "ADVInterp"
  "FeaScatter"
  "Sense"
  # "JARN_AT"
  # "Dynamic"
  # "AWP"
  # "Overfitting"
  # "ATHE"
  # "PreTrain"
  # "SAT"
  # "RobustWRN"
)
for d in "${defence[@]}"
  do
    for a in "${attacks[@]}"
    do
      echo $d $a
      job_name=${d}_${a}
      sbatch --partition gpgpu --qos gpgpumse \
             --mem=16G --gres=gpu:1 \
             --job-name $job_name --time=72:00:00\
             attack.slurm $d $a
    done
done

# # Submmit All
# declare -a defence=(
#   # "UAT"
# )
# for d in "${defence[@]}"
#   do
#     for a in "${attacks[@]}"
#     do
#       echo $d $a
#       job_name=${d}_${a}
#       sbatch --partition gpgpu --qos gpgpumse \
#              --mem=16G --gres=gpu:4 \
#              --job-name $job_name --time=72:00:00\
#              attack_parallel.slurm $d $a
#     done
# done
