# __author__ = "Abhijeet Shrivastava"

from torchvision.models.mobilenet import MobileNetV2

exp_no = 3
head = MobileNetV2().features
flatten = False
emb_dropout = 0.5
emb_dim = 128
pre_emb_param = 1280

debug = True
resume = False
model_path = 'models/model_1.pkl'
train_batch_size = 2
val_batch_size = 1
num_dataloader_worker = 1
epochs = 20
patience = 5
triplet_margin = 0.3
lr = 1e-3
eps = 1e-8
eval_interval = 1

data_dir = 'celeba'
hard_sampling = False
selector_batch_size = 2  # should be greater than or equals to 2 as batch norm is applied only if batch_norm is present

# grad clipping

