print("Hello World!")
print("Hello")

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

with tf.device('/GPU:0'):
    # Example operation
    a = tf.random.normal([10000, 10000])
    b = tf.matmul(a, a)
    print(b)

import torch
print("CUDA Available: ", torch.cuda.is_available())
print("GPU Name: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# Check if CUDA is available and then move tensor to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = torch.randn(3, 3).to(device)
    print(x)
else:
    print("CUDA is not available")
