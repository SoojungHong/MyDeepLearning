# 'batch_pos' is <class 'tensorflow.python.framework.ops.Tensor'> type 
 def tensor_to_image(self, input_tensor, img_name):
      print("debug 1 : type(input_tensor) is ", type(input_tensor)) 
      
      init_op = tf.global_variables_initializer()
      with tf.compat.v1.Session().as_default() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
          #while not coord.should_stop():
            array = sess.run(input_tensor)
            print("debug 2 : ", array)
            print("debug 3 : ", type(array))
            
            arr_ = np.squeeze(array) # you can give axis attribute if you wanna squeeze in specific dimension
            plt.imshow(arr_)
            #plt.show()
            plt.savefig(img_name+'.png')
              
            #img = Image.fromarray(array[0], "RGB") # This saved image looks wired
            #img.save(img_name+".png")
        except Exception as e:
          coord.request_stop(e)
        finally:
          coord.request_stop()
          coord.join(threads) 
    
