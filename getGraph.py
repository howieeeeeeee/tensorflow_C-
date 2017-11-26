import tensorflow as tf
import os.path
import os
import numpy as np
import inception_model_first as inception
import slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_file','/xhome/tx_zhiwei/inceptionv3_center_loss/test_pic/liang.jpg',
                           """Directory the test image location.""")
tf.app.flags.DEFINE_string('class_file','/xhome/tx_zhiwei/facial_age_images/15_16_labels.txt',
                           """dasdasdasd""")
tf.app.flags.DEFINE_string('checkpoint_dir','/xhome/tx_zhiwei/facial_age_results/freeze_test',"""where is the checkpoint""")
tf.app.flags.DEFINE_string('output_graph_dir','/xhome/tx_zhiwei/facial_age_results/freeze_test',"""where to the output_graph""")
tf.app.flags.DEFINE_string('num_classes', 89, """number of image classes + 1""")
BATCHNORM_MOVING_AVERAGE_DECAY = 0.9997

def inference(input_image, num_classes, for_training=False, restore_logits=True, scope=None):
  batch_norm_params = {
  # Decay for the moving averages.
    'decay': BATCHNORM_MOVING_AVERAGE_DECAY,
  # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
  }
  # Set weight_decay for weights in Conv and FC layers.
  with slim.arg_scope([slim.ops.conv2d, slim.ops.fc], weight_decay=0.00004):
    with slim.arg_scope([slim.ops.conv2d],
                         stddev=0.1,
                         activation=tf.nn.relu,
                         batch_norm_params=batch_norm_params):
      logits, endpoints = slim.inception.inception_v3(
                          input_image,
                          dropout_keep_prob=0.8,
                          num_classes=num_classes,
                          is_training=for_training,
                          restore_logits=restore_logits,
                          scope=scope)
      predictions = endpoints['predictions']
  return logits, predictions
  
def image_process(image_dir):
    with tf.name_scope('image_processing'):      
      filepath = image_dir
      image = tf.read_file(filepath)
      image = tf.image.decode_jpeg(image, channels=3)
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
      image = tf.image.central_crop(image, central_fraction=0.875)
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [299,299],align_corners=False)
      image = tf.squeeze(image, [0])
      image = tf.subtract(image, 0.5)
      image = tf.multiply(image, 2.0)
      image = tf.reshape(image, shape=[1, 299, 299, 3])       
      return image

def main(_):
  with tf.Graph().as_default():
    num_classes = 89
   # input_image = tf.placeholder(tf.float32, shape=(1,299,299,3),name = "input_node")
    input_image = tf.placeholder(tf.string, name = "input_node")
    image_tensor = image_process(input_image)
    logits, predictions = inference(image_tensor,FLAGS.num_classes)

    with tf.Session() as sess:
      init = tf.global_variables_initializer()
      sess.run(init)
      
      variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
      variables_to_restore = variable_averages.variables_to_restore()
      saver = tf.train.Saver(variables_to_restore)
      ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(sess, ckpt.model_checkpoint_path)
        else:
          # Restores from checkpoint with relative path.
          saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt.model_checkpoint_path))
          # Assuming model_checkpoint_path looks something like:
          #   /my-favorite-path/imagenet_train/model.ckpt-0,
          # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        print('Succesfully loaded model from %s at step=%s.' %
              (ckpt.model_checkpoint_path, global_step))
      else:
        print('No checkpoint file found')
        return

      result = sess.run(predictions, feed_dict={input_image : FLAGS.image_file})
      print(result)
      tf.train.write_graph(sess.graph, FLAGS.output_graph_dir, "age_graph.pb", as_text=True)

if __name__ =='__main__':
    tf.app.run()
