import csv

import numpy as np
from matplotlib import pyplot as plt


def draw_errorbar(init_list, emb_list, benchemark, task_name):
    num_rows = 2
    num_cols = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    for i, (init_mae_data, emb_mae_data) in enumerate(zip(init_list, emb_list)):
        ax = axes[i // num_cols, i % num_cols]
        # 计算平均值和标准差
        means_init = [np.mean(values) for key, values in init_mae_data.items()]
        stds_init = [np.std(values) for key, values in init_mae_data.items()]

        means_emb = [np.mean(values) for key, values in emb_mae_data.items()]
        stds_emb = [np.std(values) for key, values in emb_mae_data.items()]
        # 绘图
        x = np.arange(len(emb_mae_data))
        for j, key in enumerate(emb_mae_data.keys()):
            # ax.text(j, means_init[j] + stds_init[j], f"{means_init[j]:.4f}", ha='center', va='bottom', fontweight='bold')  # 输出中心点的值
            # ax.text(j, means_emb[j] + stds_emb[j], f"{means_emb[j]:.4f}", ha='center', va='bottom', fontweight='bold')  # 输出中心点的值
            ax.errorbar(j, means_init[j], yerr=stds_init[j], fmt='o', capsize=4, markersize=6,
                        color=(144 / 255, 188 / 255, 216 / 255), linewidth=2.50)
            ax.errorbar(j, means_emb[j], yerr=stds_emb[j], fmt='o', capsize=4, markersize=6,
                        color=(0.9372549019607843, 0.396078431372549, 0.403921568627451), linewidth=2.50)

        # 榜一虚线
        ax.axhline(y=benchemark[i], color='gray', linestyle='--')  # , label='Leaderboard Model MAE')
        ax.set_xticks(x)
        ax.set_xticklabels(emb_mae_data.keys())
        ax.set_xlim(-0.5, len(init_mae_data) - 0.5)
        # ax.set_xlabel('Embedding Dimension')
        # ax.set_ylabel('Mean Absolute Error (MAE)')
        ax.set_title(task_name[i])
    fig.text(0.5, 0.04, 'Embedding Dimension', ha='center', fontsize=20)
    fig.text(0.04, 0.5, 'Mean Absolute Error (MAE)', va='center', rotation='vertical', fontsize=20)
    plt.suptitle('CGCNN For 6 Tasks In Matbench', fontsize=30)
    plt.savefig("CGCNN_compare.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.clf()


def modified_csv(csv_path):
    # grep "test_MAE" matbench_orig_6task.log | awk '{printf "%s,",$4; if(NR%5==0) printf "\n"} END {print ""}'
    """
    传入获取的csv，对它按照要求修改格式
    改为dimx，mae1，mae2，mae3，mae4，mae5
    """
    # 定义要插入的keys和对应的数据列数
    keys = ['1', '2', '4', '8', '16', '32', '64', '128']

    # 读取原始CSV文件
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    # 在第一列插入对应的key
    for i in range(len(data)):
        key_index = i // 6 % len(keys)  # 计算当前所在的key索引
        data[i].insert(0, keys[key_index])
    return data


def process_cleaned_data(cleaned_data, task):
    """
    根据任务和清理后的数据处理数据，并返回新的字典
    参数：
    - cleaned_data: 清理后的数据列表
    - task: 第一行数据在清理后的数据列表中的索引（从0开始）
    返回：
    - 新的字典，包含处理后的数据
    """
    processed_dict = {}

    for i in range(task, len(cleaned_data), 6):  # 以间隔为6读取
        dim = cleaned_data[i][0]  # 获取维度信息
        mae_values = [float(val) for val in cleaned_data[i][1:]]  # 获取对应维度的MAE值列表
        processed_dict[dim] = mae_values

    return processed_dict


if __name__ == '__main__':
    # "matbench_jdft2d",  # 636 33.1918
    # "matbench_phonons",  # 1,265 28.7606
    # "matbench_dielectric",  # 4,764 0.2711
    # "matbench_log_gvrh",  # 10,987 0.0670
    # "matbench_log_kvrh",  # 10,987 0.0491
    # "matbench_perovskites"  # 1w8 0.0269

    csv_file1 = 'cgcnn_init.csv'
    mae_data1 = modified_csv(csv_file1)
    task1_init = process_cleaned_data(mae_data1, 0)
    task2_init = process_cleaned_data(mae_data1, 1)
    task3_init = process_cleaned_data(mae_data1, 2)
    task4_init = process_cleaned_data(mae_data1, 3)
    task5_init = process_cleaned_data(mae_data1, 4)
    task6_init = process_cleaned_data(mae_data1, 5)
    init_list = [task1_init, task2_init, task3_init, task4_init, task5_init, task6_init]

    csv_file2 = 'cgcnn_emb.csv'
    mae_data2 = modified_csv(csv_file2)
    task1_emb = process_cleaned_data(mae_data2, 0)
    task2_emb = process_cleaned_data(mae_data2, 1)
    task3_emb = process_cleaned_data(mae_data2, 2)
    task4_emb = process_cleaned_data(mae_data2, 3)
    task5_emb = process_cleaned_data(mae_data2, 4)
    task6_emb = process_cleaned_data(mae_data2, 5)
    emb_list = [task1_emb, task2_emb, task3_emb, task4_emb, task5_emb, task6_emb]

    benchemark_list = [33.1918, 28.7606, 0.2711, 0.0670, 0.0491, 0.0269]
    task_name = ["matbench_jdft2d", "matbench_phonons", "matbench_dielectric", "matbench_log_gvrh", "matbench_log_kvrh",
                 "matbench_perovskites"]
    draw_errorbar(init_list, emb_list, benchemark_list, task_name)
