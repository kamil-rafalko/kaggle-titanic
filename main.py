# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('data'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
train_data = pd.read_csv('data/train.csv')
train_data.head()

# %%
test_data = pd.read_csv('data/test.csv')
test_data.head()

# %%
