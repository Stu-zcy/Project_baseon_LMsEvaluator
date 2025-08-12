import os

# 获取当前目录下的所有文件
files = [f for f in os.listdir() if f.endswith('.png')]

# 对文件按字母排序（可选）
files.sort()

# 重命名文件
for i, file in enumerate(files, 1):
    # 生成新文件名，格式为 "file_序号.png"
    new_name = f"file_{i}.png"
    
    # 重命名文件
    os.rename(file, new_name)
    print(f"Renamed: {file} -> {new_name}")

print("Renaming complete.")
