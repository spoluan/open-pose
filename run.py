import argparse
import tensorflow as tf
import sys
import time
import logging
import cv2
import numpy as np
import pandas as pd
from tensorflow.contrib import slim
import vgg
from cpm import PafNet
import common
from tensblur.smoother import Smoother
from estimator import PoseEstimator, TfPoseEstimator
import threading
from threading import Thread
import queue
import os 

class MSS1407BLAB_Thread(object):
    def __init__(self, target=None, args=(), **kwargs):
        self._que = queue.Queue()
        self._t = Thread(target=lambda q,arg1,kwargs1: q.put(target(*arg1, **kwargs1)) ,
                args=(self._que, args, kwargs), )
        self._t.start()

    def join(self):
        self._t.join()
        return self._que.get()

 

logger = logging.getLogger('run')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

 
def execute(tensor_peaks, hm_up, cpm_up, img, size, image, link_video, video_saver):  
    peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                         feed_dict={raw_img: img, img_size: size})
    bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
    image = TfPoseEstimator.draw_humans(image, bodys, imgcopy=False)
    #logger.info('{}\n' . format(bodys[0]))
    #fps = round(1 / (time.time() - time_n), 2)
    #image = cv2.putText(image, str(fps)+'fps', (10, 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))
    #time_n = time.time()
    
    #if args.video is not None:
    #    image[27:img_corner.shape[0]+27, :img_corner.shape[1]] = img_corner  # [3:-10, :]
        
    #cv2.imshow('Results', image)
    if link_video is not None:
        video_saver.write(image)
        #cv2.imwrite('images/output_video/image_output_{}.jpg' . format(count), image)
        logger.infor('Frame saved!')
         

def check_(check):
    a = check.join() 

if __name__ == '__main__':
    
    tf.reset_default_graph()

    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/train/')
    parser.add_argument('--backbone_net_ckpt_path', type=str, default='checkpoints/vgg/vgg_19.ckpt')
    parser.add_argument('--image', type=str, default=None)
    # parser.add_argument('--run_model', type=str, default='img')
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--camera', type=str, default=None)
    parser.add_argument('--train_vgg', type=bool, default=True)
    parser.add_argument('--use_bn', type=bool, default=False)
    parser.add_argument('--save_video', type=str, default='images/output_video/video_output.avi')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    logger.info('checkpoint_path: ' + checkpoint_path)

    with tf.name_scope('inputs'):
        raw_img = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        img_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='original_image_size')

    img_normalized = raw_img / 255 - 0.5

    # define vgg19
    with slim.arg_scope(vgg.vgg_arg_scope()):
        vgg_outputs, end_points = vgg.vgg_19(img_normalized)

    # get net graph
    logger.info('initializing model...')
    net = PafNet(inputs_x=vgg_outputs, use_bn=args.use_bn)
    hm_pre, cpm_pre, added_layers_out = net.gen_net()

    hm_up = tf.image.resize_area(hm_pre[5], img_size)
    cpm_up = tf.image.resize_area(cpm_pre[5], img_size)
    # hm_up = hm_pre[5]
    # cpm_up = cpm_pre[5]
    smoother = Smoother({'data': hm_up}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()

    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    tensor_peaks = tf.where(tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat,
                                 tf.zeros_like(gaussian_heatMat))

    logger.info('initialize saver...')
    # trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    # trainable_var_list = []
    trainable_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='openpose_layers')
    if args.train_vgg:
        trainable_var_list = trainable_var_list + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

    restorer = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19'), name='vgg_restorer')
    saver = tf.train.Saver(trainable_var_list)
    logger.info('initialize session...')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    file_x = os.listdir('images/output_video')
    for i in file_x:
       if '.csv' in i:
          os.remove('{}/{}' . format('images/output_video', i))
          logger.info('{}/{} has been removed.' . format('images/output_video', i))

    
    
    #### Set start index frame
    count = 352 

    get = pd.DataFrame([], columns=[x.upper() for x in ['user', 
                                        'spine_base_x', 'spine_base_y', 'spine_mid_x', 'spine_mid_y',
                                        'hip_right_x', 'hip_right_y', 'hip_left_x', 'hip_left_y', 
                                        'knee_right_x', 'knee_right_y', 'knee_left_x', 'knee_left_y',
                                        'ankle_right_x', 'ankle_right_y', 'ankle_left_x', 'ankle_left_y', 
                                        'foot_right_x', 'foot_right_y', 'foot_left_x', 'foot_left_y',
                                        'hip_knee_right', 'hip_knee_left', 'knee_ankle_right', 'knee_ankle_left',
                                        'time_stamp', 'count_frame']])  
    address_file = open('images/output_video/output_video.csv', 'w')
    get.to_csv(address_file, index=False)
    address_file.close()
    time_stamp = ''
    hour = 0
    minute = 0
    second = 0
    milisecond = 0
    count_mil = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.group(tf.global_variables_initializer()))
        logger.info('restoring vgg weights...')
        restorer.restore(sess, args.backbone_net_ckpt_path)
        logger.info('restoring from checkpoint...')
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_path))
        # saver.restore(sess, args.checkpoint_path + 'model-55000.ckpt')
        logger.info('initialization done')
        if args.image is None:
            if args.video is not None:
                logger.info('Open video={}' . format(args.video))
                cap = cv2.VideoCapture(args.video)
            if args.camera is not None:
                cap = cv2.VideoCapture(0)
                # cap = cv2.VideoCapture('mplayer tv:// -tv driver=v4l2:width=640:height=480:device=/dev/video0')
            #_, image = cap.read()
            #if image is None:
                #logger.error("Can't read video, path=%s" % args.save_video)
                #sys.exit(-1)
            fps = cap.get(cv2.CAP_PROP_FPS)
            ori_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ori_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if args.save_video is not None:
                #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                fourcc = cv2.VideoWriter_fourcc('M','J','P','G') 
                #logger.info('{}, {}' . format(ori_w, ori_h))
                out = cv2.VideoWriter(args.save_video, fourcc, fps, (ori_w, ori_h))
                logger.info('record video to %s' % args.save_video)
            logger.info('fps@%f' % fps)
            # size = [int(654 * (ori_h / ori_w)), 654]
            size = [ori_w, ori_h] # (width, height)
            # h = int(654 * (ori_h / ori_w))
            time_n = time.time()  
            
            while True:
                #time_stamp = '{}:{}:{}:{}' . format(
						#'0{}' . format(hour) if len(str(hour)) == 1 else hour, 
						#'0{}' . format(minute) if len(str(minute)) == 1 else minute, 
						#'0{}' . format(second) if len(str(second)) == 1 else second, 
						#'{}' . format(milisecond) if len(str(milisecond)) == 1 else milisecond)
                try: 
                    _, image = cap.read()
                    # img = np.array(cv2.resize(image, (654, h)))
                    img = np.array(image)
                    # cv2.imshow('Original', img) 
                    #img_corner = np.array(cv2.resize(image, (360, int(360*(ori_h/ori_w)))))
                    img = img[np.newaxis, :]

                    peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up],
                                                         feed_dict={raw_img: img, img_size: size})
                    bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
                    image = TfPoseEstimator.draw_humans(image, bodys, count, imgcopy=False)
                    if args.save_video is not None:
                        out.write(image)
                        #cv2.imwrite('images/output_video/image_output_{}.jpg' . format(count), image)
                        logger.info('Frame at {} saved!' . format(count))

                    #check = MSS1407BLAB_Thread(target=execute, args=[tensor_peaks, hm_up, cpm_up, img, size, image, args.save_video, video_saver]) 
                    #threading.Thread(target=check_, args=[check]).start()

                    if args.video is not None:
                    	cv2.waitKey(1)
                    if args.camera is not None: # Used for direct camera only
                    	c = cv2.waitKey(0)
                    	if c == 27: # Wait for esc key
                            logger.info('yah noh')
                            out.release()
                            cv2.destroyAllWindows()
                            break
                 
                    #if count == 4: 
                        #out.release()
                        #cv2.destroyAllWindows()
                        #break
                    count += 1
                except Exception as ex: 
                    logger.info(ex)
                    out.release()
                    cv2.destroyAllWindows()
                    break
                # Time stamp processing
                if (milisecond == 1 or milisecond == 3 or milisecond == 7) and count_mil < 1:
                     milisecond = milisecond
                     count_mil += 1
                else:
                     milisecond += 1
                     count_mil = 0
                     if milisecond == 10:
                         milisecond = 0
                         second += 1
                         if second == 60:
                             minute += 1
                             second = 0
                             if minute == 60:
                                 hour += 1
                                 minute = 0
                
        else:
            image = common.read_imgfile(args.image)
            #image = cv2.flip(image, 1)
            size = [image.shape[0], image.shape[1]]
            logger.info(size)
            if image is None:
                logger.error('Image can not be read, path=%s' % args.image)
                sys.exit(-1)        
            h = int(654 * (size[0] / size[1]))
            img = np.array(cv2.resize(image, (654, h))) # The resize is here
            cv2.imshow('Original Image', img)
            img = img[np.newaxis, :]
            peaks, heatmap, vectormap = sess.run([tensor_peaks, hm_up, cpm_up], feed_dict={raw_img: img, img_size: size})
            #cv2.imshow('Vector mapping', vectormap[0, :, :, 0])
            bodys = PoseEstimator.estimate_paf(peaks[0], heatmap[0], vectormap[0])
            image = TfPoseEstimator.draw_humans(image, bodys, 0, imgcopy=False)
            cv2.imwrite('images/output_image/image_output.jpg', image)
            image = cv2.resize(image, (654, h))
            cv2.imshow('Results', image)
            cv2.waitKey(0)
            logger.info('Press Esc to exit')

    
