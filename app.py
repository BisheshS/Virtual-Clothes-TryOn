from flask import Flask, render_template, request, url_for
from flask_uploads import configure_uploads, UploadSet, IMAGES
import os
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for
import sys 
import numpy as np
from PIL import Image
import base64
import re
from io import BytesIO     # for handling byte strings
from io import StringIO    # for handling unicode strings
import scipy.misc
from pathlib import Path
import cv2
import shutil



def color_transfer(source, target):
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)
 
	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	# a -= aMeanTar
	# b -= bMeanTar
 
	# scale by the standard deviations
	l = (lStdTar / lStdSrc) * l
	# a = (aStdTar / aStdSrc) * a
	# b = (bStdTar / bStdSrc) * b
 
	# add in the source mean
	l += lMeanSrc
	# a += aMeanSrc
	# b += bMeanSrc
 
	# clip the pixel intensities to [0, 255] if they fall outside
	# this range
	l = np.clip(l, 0, 255)
	# a = np.clip(a, 0, 255)
	# b = np.clip(b, 0, 255)
	print("--------------in color transfer---------------")
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer


def histogram_normalization(file):
	img = cv2.imread(file)
	img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	# equalize the histogram of the Y channel
	img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
	# convert the YUV image back to RGB format
	img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
	print("--------------in histogram_normalization---------------")
	# saving_name = Path("/home/danish/Downloads/Pot/rice/static/photos/test/dumpling_first.jpg")
	cv2.imwrite(file, img)
	return




app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config.update({
    'UPLOADS_DEFAULT_DEST': os.path.realpath('.') + '/static',
    'UPLOADS_DEFAULT_URL': 'http://localhost:5000'
})

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

@app.route('/')
def index():
    return render_template('app.html')


@app.route('/save', methods=['POST'])
def get_image():
	data_url = request.values['imageBase64']
	if data_url:
		print("Image recieved")
	else:
		print("Failed-----------")
	offset = data_url.index(',')+1
	img_bytes = base64.b64decode(data_url[offset:])
	img = Image.open(BytesIO(img_bytes))
	img = np.array(img)
	loc = '/home/danish/Project/mix/static/photos/test'
	scipy.misc.imsave(os.path.join(loc,"dumpling_first.jpg"),img)
	src_dir = '/home/danish/Project/mix/static/photos/test/dumpling_first.jpg'
	final_dir = '/home/danish/Project/mix/static/dumpling_first.jpg'
	shutil.copy(src_dir,final_dir)
	histogram_normalization(os.path.join(loc,"dumpling_first.jpg"))
	return '',204


@app.route('/upload1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST' and 'photo' in request.files:
        myfile = request.files['photo']
        loc = '/home/danish/Project/mix/static/photos/test'
        saving_name = Path("/home/danish/Project/mix/static/photos/test/dumpling_first.jpg")
        if saving_name.exists():
        	os.remove(saving_name)
        photos.save(myfile,'test', 'dumpling_first'+'.'+'jpg')
        histogram_normalization(os.path.join(loc,"dumpling_first.jpg"))
        return '',204
    return '',204

@app.route('/upload2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST' and 'photo' in request.files:
        myfile = request.files['photo']
        loc = '/home/danish/Project/mix/static/photos/test'
        saving_name = Path("/home/danish/Project/mix/static/photos/test/dumpling_second.jpg")
        if saving_name.exists():
        	os.remove(saving_name)
        photos.save(myfile,'test', 'dumpling_second'+'.'+'jpg')
        histogram_normalization(os.path.join(loc,"dumpling_second.jpg"))
        return '',204
    return '',204




@app.route('/seg_one',methods=['GET','POST'])
def seg_one():
	# filename = '/home/danish/Downloads/Pot/rice/static/second_segmented.jpg'
	# Provide path for image that is uploaded in static/photos/test
	in_file = '/home/danish/Project/mix/static/photos/test/dumpling_first.jpg'
	saving_name = Path("/home/danish/Project/mix/static/first_segmented.png")
	if saving_name.exists():
		os.remove(saving_name)
	os.system('python3 first_segment.py splash --weights=mask_rcnn_shirt_0005.h5 --image="%s"' % in_file)
	print("--------------in first image---------------")
	# path and image of output image
	out_file = '/home/danish/Project/mix/static/first_segmented.png'
	return '',204



@app.route('/seg_two',methods=['GET','POST'])
def seg_two():
	# filename = '/home/danish/Downloads/Pot/rice/static/second_segmented.jpg'
	# Provide path for image that is uploaded in static/photos/test
	in_file = '/home/danish/Project/mix/static/photos/test/dumpling_second.jpg'
	saving_name = Path("/home/danish/Project/mix/static/second_segmented.png")
	if saving_name.exists():
		os.remove(saving_name)
	os.system('python3 second_segment.py splash --weights=mask_rcnn_shirt_0005.h5 --image="%s"' % in_file)
	print("--------------in second image---------------")
	# path and image of output image
	out_file = '/home/danish/Project/mix/static/second_segmented.png'
	return '',204

@app.route('/generate', methods=['GET','POST'])
def generate():
	in_file_raw1 = '/home/danish/Project/mix/static/photos/test/dumpling_first.jpg'
	in_file_raw2 = '/home/danish/Project/mix/static/photos/test/dumpling_second.jpg'
	in_file_seg1 = '/home/danish/Project/mix/static/first_segmented.png'
	in_file_seg2 = '/home/danish/Project/mix/static/second_segmented.png'
	os.system('python3 stylize.py --mask_n_colors=1 --content_img="{0}" --target_mask="{1}" --style_img="{2}" --style_mask="{3}"'.format(in_file_raw1, in_file_seg1, in_file_raw2, in_file_seg2))
	out_file = '/home/danish/Project/mix/static/result_final.png'
	return '',204


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')