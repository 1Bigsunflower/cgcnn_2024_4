import csv


def calculate_and_print_averages(input_csv):
    with open(input_csv, 'r') as infile:
        reader = csv.reader(infile)

        for row in reader:
            # 转换每行的字符串数字为浮点数
            numbers = [float(value) for value in row]
            # 计算平均值
            average = sum(numbers) / len(numbers)
            # 打印平均值
            print(average)


# 输入CSV文件名
input_csv = 'cgcnn_emb.csv'

calculate_and_print_averages(input_csv)
