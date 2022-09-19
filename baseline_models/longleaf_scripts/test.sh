#!/usr/bin/env bash


MEM=128GB
num_gpus=5
# gpu_partition=volta-gpu
gpu_partition=beta-gpu

results_dir="results_1"
search_model_path="/proj/ssrivalab/users/sayghosh/WebShop/author_models/search_il_checkpoint_800"
choice_model_path="/proj/ssrivalab/users/sayghosh/WebShop/author_models/choice_il_epoch9.pth"


OUT_LOG="${results_dir}"
mkdir -p ${OUT_LOG}

CMD=("python test.py --results_dir ${results_dir} --model_path ${choice_model_path} --bart_path ${search_model_path} --start_id 0 --end_id 99")
commands+=("srun --gres=gpu:1 --qos=gpu_access --mem=$MEM --partition=${gpu_partition} -t 00-12:00:00 \
        ${CMD} &> ${OUT_LOG}/test_1.log")
echo ${CMD} > ${OUT_LOG}/run.cmd


CMD=("python test.py --results_dir ${results_dir} --model_path ${choice_model_path} --bart_path ${search_model_path} --start_id 100 --end_id 199")
commands+=("srun --gres=gpu:1 --qos=gpu_access --mem=$MEM --partition=${gpu_partition} -t 00-12:00:00 \
        ${CMD} &> ${OUT_LOG}/test_2.log")
echo ${CMD} > ${OUT_LOG}/run.cmd

CMD=("python test.py --results_dir ${results_dir} --model_path ${choice_model_path} --bart_path ${search_model_path} --start_id 200 --end_id 299")
commands+=("srun --gres=gpu:1 --qos=gpu_access --mem=$MEM --partition=${gpu_partition} -t 00-12:00:00 \
        ${CMD} &> ${OUT_LOG}/test_3.log")
echo ${CMD} > ${OUT_LOG}/run.cmd

CMD=("python test.py --results_dir ${results_dir} --model_path ${choice_model_path} --bart_path ${search_model_path} --start_id 300 --end_id 399")
commands+=("srun --gres=gpu:1 --qos=gpu_access --mem=$MEM --partition=${gpu_partition} -t 00-12:00:00 \
        ${CMD} &> ${OUT_LOG}/test_4.log")
echo ${CMD} > ${OUT_LOG}/run.cmd

CMD=("python test.py --results_dir ${results_dir} --model_path ${choice_model_path} --bart_path ${search_model_path} --start_id 400 --end_id 499")
commands+=("srun --gres=gpu:1 --qos=gpu_access --mem=$MEM --partition=${gpu_partition} -t 00-12:00:00 \
        ${CMD} &> ${OUT_LOG}/test_5.log")
echo ${CMD} > ${OUT_LOG}/run.cmd


# now distribute them to the gpus
num_jobs=${#commands[@]}
num_gpus=$(( num_gpus < num_jobs ? num_gpus : num_jobs ))
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
        echo "Starting job $jobid on gpu $gpuid"
        # echo ${comm}
        eval ${comm}
    done &
    j=$((j + 1))
done