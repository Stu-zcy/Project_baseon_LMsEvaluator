
import os

from utils.config_parser import parse_config

import time

from utils.database_helper import extractResult
from jwt_token import sign
from user_config.config_gen import generate_config

def execute_attack(username):
    

    initTime = int(time.time())

    initTime = str(initTime)
    project_path = os.path.dirname(os.path.abspath(__file__))
    model_class = parse_config(project_path, initTime, str(username))
    model_class.run()


execute_attack("admin")