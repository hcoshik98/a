def face_encode(model,inputs,outputs):
	import face_recognition, cv2, glob, os
	from yolo_align import facealign
	import tensorflow as tf
	known_face_encodings = []
	known_face_names = []

	#run a loop on each photos kept in the images directory
	for name in glob.glob("images/*"):
		for im_path in glob.glob(name+"/*"):
			a_image = cv2.imread(im_path)
			_, align_im, _ = facealign(a_image, full=False)
			# create face encodings
			a_face_encoding, _ =model.run([outputs, {inputs: tf.convert_to_tensor(align_im)}],feed_dict={'import/phase_train:0':0, 'import/batch_size:0':1})
			print("Hi")
			# replace some words in name to write the image name
			name = name.replace('images/','')
			known_face_encodings.append(a_face_encoding)
			known_face_names.append(name)
	
	return known_face_encodings, known_face_names