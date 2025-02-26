import re
import json

def extractResult(path):
	f = open(path, "r", encoding='utf-8')
	content = iter(f.readlines())
	f.close()
	result = {k: [] for k in ['AdvAttack', 'BackDoorAttack', 'PoisoningAttack', 'SWAT']}
	# def GetPair(s: str):
	# 	ret = re.match(r'\|\s*([a-zA-Z][a-zA-Z ]*[a-zA-Z]):?\s*\|\s*([\d.]+%?)\s*\|', s)
	# 	return ret.group(1,2)
	def getSentence(s):
		ret = re.search(r'= \[(.*)\]', s)
		return ret.group(1)
	def getData(s, single=True):
		if single:
			ret = re.search(r'\s([\d\.]+%?)\s', s)
			return ret.group(1)
		else:
			ret = re.findall(r'\s([\d\.]+%?)\s', s)
			return ret
	def getContent(s):
		ret = re.search(r'\|.*\|\s*(\S*)\s*\|', s)
		return ret.group(1)
	def getRouges(s):
		ret = re.search(r'\'rouge1\': ([\d\.]+), \'rouge2\': ([\d\.]+), \'rougeL\': ([\d\.]+)', s)
		return ret.groups()
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
				result['AdvAttack'].append(ls)
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
		except StopIteration:
			break
	return result


if __name__ == "__main__":
	import os
	lmsDir = os.path.dirname(os.path.abspath(__file__))
	path = lmsDir + "\\logs\\u1h_single_1737727113_2025-01-24.txt"
	res = extractResult(path)
	j = json.dumps(res, indent=2)
	print(j)