import csv

"""
该文件利用文本对抗攻击所生成的对抗样本来构建投毒样本，即利用"attack/AdversarialAttack/log.csv"中的数据生成投毒数据
注：使用前请确保"attack/AdversarialAttack/log.csv"中含有满足投毒攻击需要的样本，本文件默认已经执行完文本对抗攻击
"""

if __name__ == "__main__":

    poisoning_data = []

    with open('../AdversarialAttack/log.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        print("header: " + str(header))
        for row in reader:
            # print(row)
            if not row[4] == row[5]:
                temp = []
                temp.append(row[1].replace("[[", "").replace("]]", ""))
                temp.append(row[5])
                temp.append(row[4])
                poisoning_data.append(temp)

    poisoning_header = ["poisoning_data", "poisoning_label", "original_label"]
    with open("poisoning_data.csv", "w", encoding='utf-8') as file:
        writer = csv.writer(file)
        for poisoning_item in poisoning_data:
            writer.writerow(poisoning_item)
