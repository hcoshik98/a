import frame
import encoding as en
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.keras.models import load_model

with tf.gfile.GFile("./20180402-114759.pb", "rb") as f:
	frozen_graph = tf.GraphDef()
	frozen_graph.ParseFromString(f.read())


# trtgraph = trt.create_inference_graph (
# 	input_graph_def = frozen_graph, outputs = "import/embeddings:0",
# 	max_batch_size = 1, max_workspace_size_bytes = 1<<30,
# 	precision_mode ='INT8',
# 	minimum_segment_size=5)


with tf.Graph().as_default() as graph:
	tf.import_graph_def(frozen_graph)
	inputs = graph.get_tensor_by_name("import/input:0")
	#inputs = [n+":0" for n in inputs] 
	outputs = graph.get_tensor_by_name("import/embeddings:0")


with tf.Session(graph=graph) as sess:
	
	[encode,name] = en.face_encode(sess,inputs,outputs)
	frame.face_loc(encode,name)
	for node in frozen_graph.node:
		if node.op == 'RefSwitch':
			node.op = 'Switch'
			for index in range(len(node.input)):
				if 'moving' in node.input[index]:
					node.input[index] = node.input[index] + '/read'
		elif node.op == 'AssignSub':
			node.op = 'Sub'
			if 'use_locking' in node.attr: del node.attr['use_locking']

	trtgraph = trt.create_inference_graph (
		input_graph_def = frozen_graph, outputs = out,
		max_batch_size = 1, max_workspace_size_bytes = 1<<30,
		precision_mode ='INT8',
		minimum_segment_size=5)


	
	

	
