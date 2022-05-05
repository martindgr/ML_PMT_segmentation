# you might want to comment out this section - specific to my machine
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# ----------------- end section -------------------------------------


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.predict import predict_multiple

dataset = 'substacks_far' 
num_epochs = 10
# check it found a GPU (assuming you are using one)
print(tf.config.list_physical_devices('GPU'))


# grab the predefined model, make sure the shape matches the images
# there are many available or you may end up making your own
model = vgg_unet(n_classes=3,input_height=416, input_width=416)
model.summary()


# you'll need to change this
base_path = f"/home/hep/dm3315/datasets/{dataset}/"
check = f"{dataset}_unet_aug_cust"


class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions()
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


# train the mode l
history = model.train(
    train_images = f"{base_path}train_frames/train/",
    train_annotations = f"{base_path}train_masks/train/",
	val_images=f"{base_path}val_frames/val/",
	val_annotations=f"{base_path}val_masks/val/",
	validate=True,
    checkpoints_path = f"/vols/t2k/users/dmartin/PMT_learning/checkpoints/{check}",
    epochs = num_epochs,
	do_augment=True,
	augmentation_name='aug_custom',
#	auto_resume_checkpoint=True,
#	load_weights=f'/vols/t2k/users/dmartin/PMT_learning/checkpoints/substacks_far_unet_aug_all_geom.29'
)


# check the predictions - use the other script (check_results.py)
# to see predictions against inputsredict_multiple(
#predict_multiple(
#    checkpoints_path = f"/home/hep/dm3315/checkpoints/{check}",
#    inp_dir =f"{base_path}test_frames/test/",
#    out_dir = f"{base_path}output/",
#	overlay_img = True,
#	callbacks = [DisplayCallback()]
#)

print('Model Evaluation:')
print(model.evaluate_segmentation(inp_images_dir=f"{base_path}val_frames/val/",
								  annotations_dir=f"{base_path}val_masks/val/"))


print('History:')
print(history.history)
# Plot training & validation accuracy values
#plt.plot(np.arange(0, num_epochs), history.history['accuracy'])
#plt.plot(np.arange(0, num_epochs), history.history['val_accuracy'])
#plt.title('Model accuracy')
#plt.ylabel('Accuracy')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#plt.savefig(f'{base_path}output/history_acc.png')

# Plot training & validation loss values
#plt.plot(np.arange(0, num_epochs), history.history['loss'])
#plt.plot(np.arange(0, num_epochs), history.history['val_loss'])
#plt.title('Model loss')
#plt.ylabel('Loss')
#plt.xlabel('Epoch')
#plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
#plt.savefig(f'{base_path}output/history_loss.png')

hist_data = open(f'{base_path}output/history_{check}_{num_epochs}_epochs.txt', 'w')
hist_data.write(f'{history.history}')
hist_data.close()
