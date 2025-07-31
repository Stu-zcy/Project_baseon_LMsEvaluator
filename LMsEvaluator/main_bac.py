import os

from utils.config_parser import parse_config

if __name__ == "__main__":
    # 下游任务执行
    projectPath = os.path.dirname(os.path.abspath(__file__))
    modelClass = parse_config(projectPath)
    modelClass.run()
