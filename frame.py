def face_loc(model, known_face_encodings, known_face_names):
	#import libraries
	import face_recognition, cv2, json
	import struct
	from yolo_align import facealign
	video_capture = cv2.VideoCapture(1)
	face_locations = []
	face_encodings = []
	face_names = []
	#To set the frame rate
	video_capture.set(3,510)
	video_capture.set(4,510)
	video_capture.set(5,45)

	while True:
		#Start reading the frame
		ret, frame = video_capture.read()
		# resize the frame by 0.25 to reduce calculataion
		small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
		rgb_small_frame = small_frame[:, :, ::-1]

		if process_this_frame:
			# gives the location of the face in terms of pixels
			face_locations, align_ims, img = facealign(a_image, full=False)
			# gives the encodings of the frame image
			face_names = []
			#loop for saving mutiple face detection
			for align_im in align_ims:
				print("Hi")
				face_encoding, _  = model.run([outputs, {inputs: tf.convert_to_tensor(align_im)}],feed_dict={'import/phase_train:0':0, 'import/batch_size:0':1})
				matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance = 0.7)
				name = "Unknown"
				if True in matches:
					first_match_index = matches.index(True)
					name = known_face_names[first_match_index]

				face_names.append(name)
				
		process_this_frame = not process_this_frame
		# open a text file to save the feature information

		for (top, right, bottom, left), name in zip(face_locations, face_names):
			top *= 4
			right *= 4
			bottom *= 4
			left *= 4

			# create a bounding rectangle to right name etc
			cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
			cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 0, 255), cv2.FILLED)
			font = cv2.FONT_HERSHEY_DUPLEX
			cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.25, (255, 255, 255), 1)
			
		i=0
		cv2.imshow('Video', frame)
		# if you press q then the loop breaks
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	cv2.destroyAllWindows()
