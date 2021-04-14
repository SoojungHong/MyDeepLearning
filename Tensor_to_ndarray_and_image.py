# 'batch_pos' is <class 'tensorflow.python.framework.ops.Tensor'> type 
init_op = tf.global_variables_initializer()
with tf.compat.v1.Session().as_default() as sess:
  sess.run(init_op)
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)
  try:
    while not coord.should_stop():
      array = sess.run(batch_pos) # Tensor to ndarray 
      print("array : ", array)
      print("type(array) : ", type(array))
              
      img = Image.fromarray(array[0], "RGB") # ndarray to image
      img.save("sj_test.png")
  except Exception as e:
    coord.request_stop(e)
  finally:
    coord.request_stop()
    coord.join(threads)
