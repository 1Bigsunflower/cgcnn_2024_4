#! /bin/bash
# nohup bash train_test.sh > train.log 2>&1 &

# 保存脚本原始工作目录
original_dir=$(pwd)

# 设置最大并行数量
max_parallel=1
# 计数器
parallel_count=0

cgcnn_path="cgcnn_lightning"
cgcnn_emb_path="cgcnn_lightning_emb"

atom_fea_len_value=(1 2 4 8 16 32 64 128)

for dim in "${atom_fea_len_value[@]}"
do
      echo "Running with atom_fea_len = $dim"

      cd "$cgcnn_path" || exit 1
      nohup python cgcnn_lightning_bu.py --atom_fea_len "$dim" > "cgcnn_dim${dim}_bu.log" 2>&1 &
      cd "$original_dir" || exit 1

      ((parallel_count++))
      if ((parallel_count >= max_parallel)); then
        # 等待任一后台进程执行完成
        wait -n
        ((parallel_count--))
        echo "1/2"
      fi

      cd "$cgcnn_emb_path" || exit 1
      nohup python cgcnn_lightning_bu.py --atom_fea_len "$dim" > "cgcnn_dim${dim}_emb_bu.log" 2>&1 &
      cd "$original_dir" || exit 1

      ((parallel_count++))
      if ((parallel_count >= max_parallel)); then
        wait -n
        ((parallel_count--))
      fi

done
echo "Finish."

