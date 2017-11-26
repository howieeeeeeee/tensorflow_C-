import tensorflow as tf
import os.path
import os
import numpy as np
import inception_model_first as inception
from sklearn.svm import SVR
from sklearn.externals import joblib

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('image_dir','/xhome/tx_zhiwei/inceptionv3_center_loss/test_pic',
                           """Directory the test image location.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/xhome/tx_zhiwei/facial_age_results/15_16_train_softmax_ldl_loss_convx2_ft_steps_100000',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_string('class_file','/xhome/tx_zhiwei/facial_age_images/15_16_labels.txt',
                           """dasdasdasd""")

def _eval_once(saver,endpoints):
    with tf.device('/cpu:0'):
        with tf.Session() as sess:
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
            result = sess.run(endpoints)
            
            #test_fea = np.array(result['fc'])
            
            #svr_model = joblib.load('/xhome/tx_zhiwei/inceptionv3/inception/svr_model.pkl') 
            #predict_age = np.rint( svr_model.predict(test_fea) )
            #predict_age = predict_age.astype(np.int64)
            
            
            result = np.array(result['predictions'])
            predict_label = np.argmax(result, axis=1)  
            

            with open(FLAGS.class_file,'r+') as f:
                label=f.readlines()
            print(label[0])
            for i in predict_label:
                print('cnn', label[i-1])
            #for i in predict_age:
            #    print('svr', label[i])

def image_process(image_dir):
    with tf.name_scope('image_processing'):
        list = os.listdir(image_dir)
        
        for i in list:
          print(i)
        
        filepath = os.path.join(image_dir, list[0])
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
        for line in list[1:]:
          filepath = os.path.join(image_dir, line)
          image_n = tf.read_file(filepath)
          image_n = tf.image.decode_jpeg(image_n, channels=3)
          image_n = tf.image.convert_image_dtype(image_n, dtype=tf.float32)
          image_n = tf.image.central_crop(image_n, central_fraction=0.875)
          image_n = tf.expand_dims(image_n, 0)
          image_n = tf.image.resize_bilinear(image_n, [299,299],align_corners=False)
          image_n = tf.squeeze(image_n, [0])
          image_n = tf.subtract(image_n, 0.5)
          image_n = tf.multiply(image_n, 2.0)
          image_n = tf.reshape(image_n, shape=[1, 299, 299, 3])
          image = tf.concat([image, image_n], 0)
        
    return image

def inferance(image_dir):
    with tf.device('/cpu:0'):
        with tf.Graph().as_default():
            image = image_process(image_dir)
            
            num_classes = 89
            logits, _, endpoints = inception.inference(image, num_classes)
            variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            # summary_op = tf.summary.merge_all() 
            #graph_def = tf.get_default_graph().as_graph_def()
            #summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,graph_def=graph_def)
            _eval_once(saver,endpoints)

def main(_):
    inferance(FLAGS.image_dir)

if __name__ =='__main__':
    tf.app.run()
