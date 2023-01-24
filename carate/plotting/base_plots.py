"""
Plotting module for PyTorch prototyping

:author: Julian M. Kleber
"""
from typing import Type, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


N = 21
x = np.linspace(0, 10, 11)
y = [3.9, 4.4, 10.8, 10.3, 11.2, 13.1, 14.1, 9.9, 13.9, 15.1, 12.5]

# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)
y_est = a * x + b
y_err = x.std() * np.sqrt(
    1 / len(x) + (x - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
)

fig, ax = plt.subplots()
ax.plot(x, y_est, "-")
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(x, y, "o", color="tab:brown")

if __name__ == "__main__":
    from amarium.utils import load_json_from_file
    import pandas as pd

    result = load_json_from_file(
        "/home/dev/carate/carate/carate_config_files/Classification/PROTEINS_10/data/CV_0/PROTEINS_Epoch_4980.json"
    )
    df = pd.DataFrame(result)
    print(df.head(5))
