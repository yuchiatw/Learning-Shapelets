import numpy as np
import matplotlib.pyplot as plt
from pyts.datasets import load_gunpoint
from pyts.transformation import ShapeletTransform
from synthetic.synthetic_preprocessing import synthetic_pipeline
import time
import os
# Toy dataset
num_samples = 1000
time_step = 100
data_path = os.path.join('./data',f'synthetic_{num_samples}_{time_step}.npz')

if os.path.exists(data_path):
    data = np.load(data_path)
else:
    data = synthetic_pipeline(
        time_step=time_step,
        num_samples=num_samples,
        root='./data'
    )
X_train = data['X_train']
y_train = data['y_train']
X_train = X_train.reshape(X_train.shape[0], -1)  # Flatten the data if necessary
# Shapelet transformation
t1 = time.time()
st = ShapeletTransform(window_sizes='auto',
                       random_state=42, sort=True)
X_new = st.fit_transform(X_train, y_train)
print(time.time() - t1, "seconds for shapelet transformation")

# # Visualize the four most discriminative shapelets
# plt.figure(figsize=(6, 4))
# for i, index in enumerate(st.indices_[:4]):
#     idx, start, end = index
#     plt.plot(X_train[idx], color='C{}'.format(i),
#              label='Sample {}'.format(idx))
#     plt.plot(np.arange(start, end), X_train[idx, start:end],
#              lw=5, color='C{}'.format(i))

# plt.xlabel('Time', fontsize=12)
# plt.title('The four most discriminative shapelets', fontsize=14)
# plt.legend(loc='best', fontsize=8)
# plt.show()