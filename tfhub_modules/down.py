import tensorflow_hub as hub

# 下载模型并缓存到本地
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")

print("模型已下载到本地缓存路径！")