# model_name=chatglm2-6b
model_name=Llama2-Chinese-7b-Chat
# model_name=Qwen-7B-Chat
# model_name=Yi-1.5-9B-Chat
# model_name=internlm2-chat-20b


model_dir=/home/azureuser/weimin/models/
local_dir=checkpoints
# split=validate
split=test

save_path=outputs/${model_name}
output_path=${save_path}/output/${split}
logs_path=${save_path}/logs

if [ -d ${logs_path} ]; then
    rm -r ${logs_path}
fi
mkdir -p ${save_path}
mkdir -p ${logs_path}
mkdir -p ${output_path}


sample_num_workers=8
worker_num=64

if [ ! -d $local_dir ]; then
    mkdir $local_dir
fi

# link the model
for ((j=0;j<${sample_num_workers};j=j+1)); do
    if [ ! -d $local_dir/$model_name-$j ]; then
        echo "Model not found in $local_dir/$model_name-$j"
        ln -s ${model_dir}/${model_name} ${local_dir}/${model_name}-${j}
    fi
done

# lanuch the model
fs_worker_port=21012
worker_idx=0
for ((j=0;j<${sample_num_workers};j=j+1)); do
    echo "Launch the model worker on port ${fs_worker_port}"
    CUDA_VISIBLE_DEVICES=$((${worker_idx} % ${sample_num_workers})) python -u -m fastchat.serve.vllm_worker \
        --model-path ${local_dir}/${model_name}-${j} \
        --port ${fs_worker_port} \
        --controller-address http://localhost:21002 \
        --worker-address http://localhost:${fs_worker_port} >> ${logs_path}/model_worker-${j}.log 2>&1 &
    echo $! >> ${logs_path}/worker_pid.txt
    fs_worker_port=$(($fs_worker_port+1))
    worker_idx=$(($worker_idx+1))
    sleep 15
done

sleep 60

# generate results
for ((i=0;i<${worker_num};i=i+1)); do
    # echo "Start generating results for worker $i"
    python3 generate_results.py --model_name ${model_name}-$((i%sample_num_workers)) --split ${split} --part_num ${worker_num} --part_idx ${i} --save_path ${output_path} >> ${logs_path}/generate_results-${i}.log 2>&1 &
    echo $! >> ${logs_path}/eval_pid.txt
done

wait $(cat ${logs_path}/eval_pid.txt)
rm ${logs_path}/eval_pid.txt
echo "Exploration Finished"

# if failed, exit
if [ $? -ne 0 ]; then
    echo "Exploration failed"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt
    exit 1
fi

python gather_results.py --model_name ${model_name} --split ${split} --data_path ${output_path}

split=validate
output_path=${save_path}/output/${split}
mkdir -p ${output_path}

# generate results
for ((i=0;i<${worker_num};i=i+1)); do
    # echo "Start generating results for worker $i"
    python3 generate_results.py --model_name ${model_name}-$((i%sample_num_workers)) --split ${split} --part_num ${worker_num} --part_idx ${i} --save_path ${output_path} >> ${logs_path}/generate_results-${i}.log 2>&1 &
    echo $! >> ${logs_path}/eval_pid.txt
done

python gather_results.py --model_name ${model_name} --split ${split} --data_path ${output_path}


wait $(cat ${logs_path}/eval_pid.txt)
rm ${logs_path}/eval_pid.txt
echo "Exploration Finished"

# if failed, exit
if [ $? -ne 0 ]; then
    echo "Exploration failed"
    kill -9 $(cat ${logs_path}/worker_pid.txt)
    rm ${logs_path}/worker_pid.txt
    exit 1
fi

# kill the model worker
echo "Kill the model workers"
kill -9 $(cat ${logs_path}/worker_pid.txt)
rm ${logs_path}/worker_pid.txt