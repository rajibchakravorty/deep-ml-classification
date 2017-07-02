

data_file = '../prepare_data/cifar-10.npz'

image_height = 26
image_width  = 26
image_channel = 3
output_length = 10


batch_size = 256
num_epochs = 20000


learning_rate = 0.01
momentum = 0.95
l1_regu = 0.
l2_regu = 0.

model_file = 'cifar-10-model.npz'
init_model = model_file
start_loss = 0.78


stat_file = 'learn_stat.csv'
