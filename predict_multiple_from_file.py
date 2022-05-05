


from keras_segmentation.models.unet import vgg_unet
import tensorflow as tf
from tensorflow import keras
from keras_segmentation.predict import predict_multiple
from keras_segmentation.predict import visualize_segmentation
from keras_segmentation.predict import evaluate

import model
import os


#os.environ["CUDA_VISIBLE_DEVICES"]="1"
print(tf.config.list_physical_devices('GPU'))


#checkpoint = 'substacks_unet_aug_init_bpbaug19.9'
#checkpoint = 'substacks_unet_aug_flips_scales_20init.9'
checkpoint = 'substacks_far_unet.19'
binary_colour = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
bgr_colour = [(0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0)]

dataset = 'substacks_far'

#model = vgg_unet(n_classes=3 ,  input_height=416, input_width=416)
model = vgg_unet(n_classes=3 ,  input_height=int(2752), input_width=int(4000))
#model = vgg_unet(n_classes=3 ,  input_height=2016, input_width=3008)
#check = 'vgg_unet_1'

#data_path =  "/vols/t2k/users/dmartin/PMT_learning/Raw/"
#data_out = f'/home/hep/dm3315/datasets/{dataset}/output/PD3_aug/'
data_path =  "/vols/t2k/users/dmartin/PMT_learning/Raw/ring_avg_all/"
data_out = '/vols/t2k/users/dmartin/PMT_learning/output/ring_avg_all/'


#mypath = "/home/hep/dm3315/datasets/bolts_pmt/"
work_dir =  '/home/hep/dm3315/'


model.load_weights(f'/vols/t2k/users/dmartin/PMT_learning/checkpoints/{checkpoint}')



#get filenames
f = []
for (dirpath, dirnames, filenames) in os.walk(f'{data_path}'):
	    f.extend(filenames)
	    break

filetype = f[0][-4:]
f = [x[:-4] for x in f]

#f = ['239', '239_udist']

def predict_single(ifile, ofile, name):

	model.predict_segmentation(
    	inp= f'{ifile}{name}{filetype}',
     	out_fname=f"{ofile}{name}_pred_{checkpoint}.png",
	 	#overlay_img=True,
	 	colors = bgr_colour
		)

#predict_single('SING0046')
img_num = 0
for i in f :
	img_num += 1
	print(f"Processing image {img_num} of {len(f)} : {i}")
	predict_single(data_path, data_out, str(i))
