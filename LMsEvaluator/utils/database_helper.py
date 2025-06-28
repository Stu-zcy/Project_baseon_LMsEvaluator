import os
import re
import json
import sys
lmsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def extractResult(path):
	f = open(path, "r", encoding='utf-8')
	content = iter(f.readlines())
	f.close()
	result = {k: [] for k in ['AdversarialAttack', 'BackDoorAttack', 'PoisoningAttack', 'RLMI', 'SWAT', 'ModelStealingAttack']}
	# def GetPair(s: str):
	# 	ret = re.match(r'\|\s*([a-zA-Z][a-zA-Z ]*[a-zA-Z]):?\s*\|\s*([\d.]+%?)\s*\|', s)
	# 	return ret.group(1,2)
	def getSentence(s):
		ret = re.search(r'= \[(.*)\]', s)
		return ret.group(1)
	
	'''提取小数或整数，可带百分号
	   single=True时返回第一个匹配的结果，single=False时返回所有匹配的结果'''
	def getData(s, single=True):
		if single:
			ret = re.search(r'\s([\d\.]+%?)[\s,]', s)
			return ret.group(1)
		else:
			ret = re.findall(r'\s([\d\.]+%?)[\s,]', s)
			return ret
		
	'''获取表格化数据，会连带获取百分号'''
	def getContent(s):
		ret = re.search(r'\|.*\|\s*(\S*)\s*\|', s)
		return ret.group(1)
	
	'''专用：获取rouge分数'''
	def getRouges(s):
		ret = re.search(r'\'rouge1\': ([\d\.]+), \'rouge2\': ([\d\.]+), \'rougeL\': ([\d\.]+)', s)
		return ret.groups()
	
	'''迭代指定次数'''
	def iterate(num):
		for _ in range(num-1):
			next(content)
		return next(content)

	while True:
		try:
			row = next(content)
			if re.match(r'\|\s*Attack Results.*\|.*\|', row):
				rowNum = 6
				ls = []
				next(content)
				for _ in range(rowNum):
					row = next(content)
					ls.append(getContent(row))
				result['AdversarialAttack'].append(ls)
			elif (re.search('PoisoningAttack 攻击开始', row)):
				while not (re.match(r'\|.*Result.*\|.*\|', row)):
					row = next(content)
				rowNum = 2
				ls = []
				next(content)
				for _ in range(rowNum):
					row = next(content)
					ls.append(getContent(row))
				result['PoisoningAttack'].append(ls)
			elif (re.search('BackDoorAttack 攻击开始', row)):
				# row = iterate(2)
				# epochs = int(getData(row))
				# iterate(3)
				# ls = []
				# for i in range(epochs):
				# 	subList = ["epoch:" + str(i)]
				# 	row = next(content)
				# 	subList.append(getData(row))
				# 	row = iterate(3)
				# 	subList.append(getData(row))
				# 	row = iterate(3)
				# 	subList.append(getData(row))
				# 	ls.append(subList)
				# #training finished
				# subList = ['/', '/']
				# row = iterate(5)
				# subList.append(getData(row))
				# row = iterate(3)
				# subList.append(getData(row))
				# ls.append(subList)
				# result['BackDoorAttack'].append(ls)
				while not (re.match(r'\|.*Result.*\|.*\|', row)):
					row = next(content)
				rowNum = 7
				ls = []
				next(content)
				for _ in range(rowNum):
					row = next(content)
					ls.append(getContent(row))
				result['BackDoorAttack'].append(ls)
			elif re.match(r'\|\s*RLMI Attack Results.*\|.*\|', row):
				rowNum = 4
				ls = []
				next(content)
				for _ in range(rowNum):
					row = next(content)
					ls.append(getContent(row))
				result['RLMI'].append(ls)
			elif (re.search('SWAT 攻击开始', row)):
				row = next(content)
				ret = re.search(r'n_attacks=(\d)', row)
				n_attacks = eval(ret.group(1))
				attackList = []
				iterate(1)
				for _ in range(n_attacks):
					subList = []
					row = iterate(2)
					subList.append(getSentence(row))
					row = next(content)
					subList.append(getSentence(row))
					row = iterate(3)
					subList.extend(getRouges(row))
					row = next(content)
					subList.append(getData(row))
					row = next(content)
					subList.append(getData(row))
					row = next(content)
					subList.append(True if (re.search('True', row)) else False)
					attackList.append(subList)
				row = iterate(2)
				subList = []
				subList.extend(getRouges(row))
				for _ in range(3):
					row = next(content)
					subList.append(getData(row))
				attackList.append(subList)
				result['SWAT'].append(attackList)
			elif (re.search('ModelStealingAttack攻击开始', row)):
				# 采用了字典的方式进行存储和通讯。
				# 一个配置的结果为一个字典，包含以下键值对：
				# {iteration: [(0.4808, 0.8625, 0.9000), ...], victim_acc: 0.8862, steal_acc: 0.5020, agreement: 0.5284}
				# 其中iteration为一个列表，每一个元组是每个epoch的训练数据，分别是train_loss, train_acc(训练集准确率), vali_acc(验证集准确率)
				# victim_acc为受害者模型的准确率，steal_acc为攻击者模型的准确率，agreement为两者的一致性。

				while not (re.search('(it: 0)|(ModelStealingAttack攻击结束)', row)):
					row = next(content)
				if re.search('ModelStealingAttack攻击结束', row):
					print("ModelStealingAttack没有攻击结果?")
					continue

				row = next(content)
				match = re.search('Epoch \[\d/(\d+)\]', row)
				# 获取epoch数量
				epochNum = eval(match.group(1))
				
				localResult = {'iteration': [], 'victim_acc': 0, 'steal_acc': 0, 'agreement': 0}
				# 获取每一轮的训练数据
				for i in range(epochNum):
					localResult['iteration'].append(tuple(getData(row, single=False)))
					row = iterate(2)
				
				while not (re.search('Evaluation Result on IMDB: victim acc', row)):
					row = next(content)
				# 获取victim acc, steal acc, agreement
				data = getData(row, single=False)
				localResult['victim_acc'] = data[0]
				localResult['steal_acc'] = data[1]
				localResult['agreement'] = data[2]

				result['ModelStealingAttack'].append(localResult)
		except StopIteration:
			break
	return result

# def pushIntoDB(filename: str):
# 	info = filename.split('_')
# 	username, initTime = info[0], eval(info[2])
# 	result = extractResult(lmsDir + "\\..\\logs\\" + filename)
# 	add_attack_record(username, json.dumps(result))
# 	# attack = AttackRecord(createUserName=username, createTime=initTime, attackResult=json.dumps(result))
# 	# db.session.add(attack)
#   # db.session.commit()

if __name__ == "__main__":
	print(lmsDir)
	path = os.path.join(lmsDir, "logs", "u1h_single_1737727113_2025-01-24.txt")
	print(path)
	res = extractResult(path)
	print(res)