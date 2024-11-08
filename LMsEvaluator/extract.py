import re
import json

def extractResult(path):
	f = open(path, "r", encoding='utf-8')p	content = iter(f.readlines())
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
	def iterate(num):
		for _ in range(num-1):
			next(content)
		return next(content)

	while True:
		try:
			row = next(content)
			if re.match(r'\|.*Result.*\|.*\|', row):
				if (re.match(r'\|.*Attack Results.*\|.*\|', row)):
					rowNum = 6
				elif(re.match(r'\|\sResults\s{2}\|\s*\|', row)):
					continue
				else:
					rowNum = 2
				ls = []
				next(content)
				for _ in range(rowNum):
					row = next(content)
					ls.append(getData(row))
				result['PoisoningAttack' if rowNum == 2 else 'AdvAttack'].append(ls)
			elif (re.search('BackDoorAttack 攻击开始', row)):
				row = iterate(2)
				epochs = int(getData(row))
				iterate(3)
				ls = []
				for i in range(epochs):
					subList = ["epoch:" + str(i)]
					row = next(content)
					subList.append(getData(row))
					row = iterate(3)
					subList.append(getData(row))
					row = iterate(3)
					subList.append(getData(row))
					ls.append(subList)
				#training finished
				subList = ['/', '/']
				row = iterate(5)
				subList.append(getData(row))
				row = iterate(3)
				subList.append(getData(row))
				ls.append(subList)
				result['BackDoorAttack'].append(ls)
			elif (re.search('SWAT 攻击开始', row)):
				row = iterate(2)
				ret = re.search(r'n_attacks=(\d)', row)
				n_attacks = ret.group(1)
				attackList = []
				ls = []
				iterate(1)
				for _ in range(n_attacks):
					subList = []
					row = iterate(2)
					subList.append(getSentence(row))
					row = next(content)
					subList.append(getSentence(row))
					row = iterate(3)
					subList.extend(getData(row, False))
					row = next(content)
					subList.append(getData(row))
					row = next(content)
					subList.append(getData(row))
					row = next(content)
					subList.append(True if (re.search('True', row)) else False)
					attackList.append(subList)
				row = iterate(2)
				ls.extend(getData(row, False))
				for _ in range(3):
					row = next(content)
					ls.append(getData(row))
				ls.append(attackList)
				result['SWAT'].append(ls)
		except StopIteration:
			break
	return result










if __name__ == "__main__":
	path="./logs/single_2024-04-25.txt"
	res = extractResult(path)
	j = json.dumps(res, indent=2)
	print(j)