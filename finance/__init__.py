"""
CONFIDENTIAL
__________________
2022 Happy Health Incorporated
All Rights Reserved.
NOTICE:  All information contained herein is, and remains
the property of Happy Health Incorporated and its suppliers,
if any.  The intellectual and technical concepts contained
herein are proprietary to Happy Health Incorporated
and its suppliers and may be covered by U.S. and Foreign Patents,
patents in process, and are protected by trade secret or copyright law.
Dissemination of this information or reproduction of this material
is strictly forbidden unless prior written permission is obtained
from Happy Health Incorporated.
Authors: Lucas Selig <lucas@happy.ai>
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob, os, random
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
np.random.seed(0)
print(sys.path)
sns.set_style("darkgrid")
