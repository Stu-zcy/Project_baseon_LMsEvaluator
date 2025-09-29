from absl import logging

import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns

module_url = "https://tfhub.dev/google/universal-sentence-encoder/3"
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)