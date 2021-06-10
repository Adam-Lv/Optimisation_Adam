# Parameters of construction of network

# the size of convolution kernel
kernel_size = 5

# the number of kernels in the first convolution layer
kernel_nums1 = 8

# the number of kernels in the second convolution layer
kernel_nums2 = 16

# padding schema of convolution
padding = 'same'

# strides of the kernel
strides = 1

# the size of pooling kernel
pooling_size = 2

# the size of the first dense layer
dense_size = 128

# dropping rate
drop_rate = 0.001

# the number of labels in the output layer
label_nums = 10

# loss function
loss_func = 'sparse_categorical_crossentropy'

# times the training runs
epochs = 10

# batch size
batch_size = 32

# Parameters of Adam optimizer
learning_rate = 0.009
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
