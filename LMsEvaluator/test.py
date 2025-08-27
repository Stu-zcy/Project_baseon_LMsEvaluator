from pathlib import Path

def transfer_log_range(x1: int, x2: int, dest_file: str, src_file: str) -> int:
    """
    将 src_file 中第 x1~x2 行复制到 dest_file（1 起始、闭区间）。
    - 若 dest_file 所在目录不存在，会自动创建
    - 若 dest_file 已存在，采用追加写入
    - 若 x2 超过文件实际行数，只会复制到文件末尾
    返回：成功写入的行数
    """
    if x1 < 1 or x2 < x1:
        raise ValueError("参数不合法：要求 x1 >= 1 且 x2 >= x1")

    src = Path(src_file)
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在：{src}")

    dest = Path(dest_file)
    dest.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    # 读取时宽容处理编码问题；写入保持 UTF-8
    with src.open("r", encoding="utf-8", errors="replace") as fin, \
         dest.open("a", encoding="utf-8", newline="") as fout:
        for lineno, line in enumerate(fin, start=1):
            if lineno < x1:
                continue
            if lineno > x2:
                break
            fout.write(line)
            written += 1

    return written