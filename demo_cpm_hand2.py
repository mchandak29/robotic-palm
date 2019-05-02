#Imported Functions
import tensorflow as tf
from models.nets import cpm_hand_slim
import numpy as np
from utils import cpm_utils
import cv2
import time
import math
import sys
import tensorflow.contrib.slim as slim
import os
import csv
import numpy as np
import serial

#Define port for transmitting data serially to Arduino.
port = '/dev/ttyACM0'
s = serial.Serial(port,115200)

#??
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Defining Input File Arguments 
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('DEMO_TYPE',
                           #default='test_imgs/longhand.jpg',
                           default='SINGLE',
                           help='MULTI: show multiple stage,'
                                     'SINGLE: only last stage,'
                                     'HM: show last stage heatmap,'
                                     'paths to .jpg or .png image')
tf.app.flags.DEFINE_string('model_path',
                           default='models/weights/cpm_hand.pkl',
                           help='Your model')
tf.app.flags.DEFINE_integer('input_size',
                            default=368,
                            help='Input image size')
tf.app.flags.DEFINE_integer('hmap_size',
                            default=46,
                            help='Output heatmap size')
tf.app.flags.DEFINE_integer('cmap_radius',
                            default=21,
                            help='Center map gaussian variance')
tf.app.flags.DEFINE_integer('joints',
                            default=21,
                            help='Number of joints')
tf.app.flags.DEFINE_integer('stages',
                            default=6,
                            help='How many CPM stages')
tf.app.flags.DEFINE_integer('cam_num',
                            default=0,
                            help='Webcam device number')
tf.app.flags.DEFINE_bool('KALMAN_ON',
                         default=True,
                         help='enalbe kalman filter')
tf.app.flags.DEFINE_float('kalman_noise',
                            default=3e-2,
                            help='Kalman filter noise value')
tf.app.flags.DEFINE_string('color_channel',
                           default='RGB',
                           help='')

# Set color for each finger
joint_color_code = [[139, 53, 255],
                    [0, 56, 255],
                    [43, 140, 237],
                    [37, 168, 36],
                    [147, 147, 0],
                    [70, 17, 145]]

# Each Element of the list defines a joint to joint connection.
limbs = [[0, 1],
         [1, 2],
         [2, 3],
         [3, 4],
         [0, 5],
         [5, 6],
         [6, 7],
         [7, 8],
         [0, 9],
         [9, 10],
         [10, 11],
         [11, 12],
         [0, 13],
         [13, 14],
         [14, 15],
         [15, 16],
         [0, 17],
         [17, 18],
         [18, 19],
         [19, 20]
         ]

#Define a nx6 2-d array to store n data timesteps: angles of (index, middle, ring, little, palm, thumb).
m_avg=np.zeros((15,6))

#Refernce values to be set each time in the beginning of the execution. 
ratios=np.zeros((6))
avg_i=0


def main(argv):
    global ratios, avg_i, m_avg, s
    tf_device = '/gpu:0'
    with tf.device(tf_device):
        #Build graph
        
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 3],name='input_image')
        center_map = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_size, FLAGS.input_size, 1],name='center_map')
        model = cpm_hand_slim.CPM_Model(FLAGS.stages, FLAGS.joints + 1)
        model.build_model(input_data, center_map, 1)

    saver = tf.train.Saver()

    #Create session and restore weights
    
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=False))
    sess.run(tf.global_variables_initializer())
    model.load_weights_from_file(FLAGS.model_path, sess, False)

    #???
    test_center_map = cpm_utils.gaussian_img(FLAGS.input_size, FLAGS.input_size, FLAGS.input_size / 2,
                                             FLAGS.input_size / 2,
                                             FLAGS.cmap_radius)
    test_center_map = np.reshape(test_center_map, [1, FLAGS.input_size, FLAGS.input_size, 1])

    #Starting Video Input
    if not FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
        cam = cv2.VideoCapture(FLAGS.cam_num)

    # Create kalman filters
    kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.joints)]
    for _, joint_kalman_filter in enumerate(kalman_filter_array):
        joint_kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                        np.float32)
        joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                       np.float32) * FLAGS.kalman_noise

    with tf.device(tf_device):
        
        t0=time.time()
        
        while True:
            print(time.time()-t0)
            t1 = time.time()
            
            #Read image and resize it according to architecture input size.
            if FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
                test_img = cpm_utils.read_image(FLAGS.DEMO_TYPE, [], FLAGS.input_size, 'IMAGE')
            else:
                test_img = cpm_utils.read_image([], cam, FLAGS.input_size, 'WEBCAM')

            test_img_resize = cv2.resize(test_img, (FLAGS.input_size, FLAGS.input_size))
            
            test_img_input = test_img_resize / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)


            if FLAGS.DEMO_TYPE.endswith(('png', 'jpg')):
                ### NOT COMMENTED: TO BE REMOVED
                ### NOT COMMENTED: TO BE REMOVED
                ### NOT COMMENTED: TO BE REMOVED
                ### NOT COMMENTED: TO BE REMOVED

                # Inference
                t1 = time.time()
                predict_heatmap, stage_heatmap_np = sess.run([model.current_heatmap,
                                                              model.stage_heatmap,
                                                              ],
                                                             feed_dict={'input_image:0': test_img_input,
                                                                        'center_map:0': test_center_map})

                # Show visualized image
                demo_img,coords = visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array)
                cv2.imshow('demo_img', demo_img.astype(np.uint8))
                '''
                filedem = open("testfile.txt","w")
                for i in [0,5,7,9,11,13,15,17,19]:
                    for j in [0,1]:
                        filedem.write(str(coords[i][j])+' ')
                filedem.close()
                '''
                with open('test_file.csv', mode='w') as file:
                    writer = csv.writer(file, delimiter=',')
                    for i in [1,5,8,9,12,13,16,17,20]:
                        writer.writerow([coords[i][0],coords[i][1]])

                if cv2.waitKey(0) == ord('q'): break
                #print('fps: %.2f' % (1 / (time.time() - t1)))
                

            elif FLAGS.DEMO_TYPE == 'SINGLE':

                # Starting time of each iteration.
                t1 = time.time()

                #Run the image through the model and get corresponding heatmap.
                stage_heatmap_np = sess.run([model.stage_heatmap[2]],
                                            feed_dict={'input_image:0': test_img_input,
                                                       'center_map:0': test_center_map})
                
                #Get the coordinates of each joint and print the image with joints makrked on the figure.
                demo_img,coords = visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array)
                cv2.imshow('current heatmap', (demo_img).astype(np.uint8))    
                
                #At the beginning of each execution, user gets 30 sec 
                #for setting reference angles when hand was kept straight.
                if ((time.time()-t0)<30):
                    store_deb(coords, t0)
                    print('ratios: '+str(ratios))
                else:
                    debug(coords)
                    print('degrees: '+str(np.mean(m_avg,axis=0)))

                #Calculate moving average for past n timesteps.    
                mavg_deg=np.mean(m_avg,axis=0)
                mavg_deg[4]= min(mavg_deg[4],20)               

                avg_i=(avg_i+1)%15

                #Setting the string to be transmitted to Arduino.
                transp_str=''
                for i in mavg_deg:
                    transp_str=transp_str+str(i)+','

                #Flushing the input and output buffer before writing to Serial.
                s.flushInput()
                s.flushOutput()
                s.write(transp_str.encode())    
                

                if cv2.waitKey(1) == ord('q'): break
                            
        

def visualize_result(test_img, FLAGS, stage_heatmap_np, kalman_filter_array):
    t1 = time.time()
    demo_stage_heatmaps = []
    
    last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.joints].reshape(
        (FLAGS.hmap_size, FLAGS.hmap_size, FLAGS.joints))
    last_heatmap = cv2.resize(last_heatmap, (test_img.shape[1], test_img.shape[0]))
    #print('hm resize time %f' % (time.time() - t1))

    t1 = time.time()
    joint_coord_set = np.zeros((FLAGS.joints, 2))

    # Plot joint colors
    for joint_num in range(FLAGS.joints):
        joint_coord = np.unravel_index(np.argmax(last_heatmap[:, :, joint_num]),
                                       (test_img.shape[0], test_img.shape[1]))
        joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
        kalman_filter_array[joint_num].correct(joint_coord)
        kalman_pred = kalman_filter_array[joint_num].predict()
        joint_coord_set[joint_num, :] = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
        else:
            joint_color = list(map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            
            cv2.circle(test_img, center=(joint_coord[1], joint_coord[0]), radius=3, color=joint_color, thickness=-1)
    #print('plot joint time %f' % (time.time() - t1))

    t1 = time.time()
    # Plot limb colors
    for limb_num in range(len(limbs)):

        x1 = joint_coord_set[limbs[limb_num][0], 0]
        y1 = joint_coord_set[limbs[limb_num][0], 1]
        x2 = joint_coord_set[limbs[limb_num][1], 0]
        y2 = joint_coord_set[limbs[limb_num][1], 1]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4

            limb_color = list(map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))

            cv2.fillConvexPoly(test_img, polygon, color=limb_color)
    #print('plot limb time %f' % (time.time() - t1))
    return test_img,joint_coord_set


def debug(coords):
    #This Func. calculates the required angles.

    global m_avg
    global ratios
    
    degs=[]
    #Get the coordinates of the required joints.
    coords1 = []
    for i in [1,5,8,9,12,13,16,17,20,0]:
        coords1.append(coords[i])
    #Get median of all joints
    Xc1=(coords1[0][0]+coords1[1][0]+coords1[3][0]+coords1[5][0]+coords1[7][0])/5
    Yc1=(coords1[0][1]+coords1[1][1]+coords1[3][1]+coords1[5][1]+coords1[7][1])/5
    
    '''Get the ratios of 4 fingers to calculate finger angles.
       ratio of length of finger to distance between finger-palm joint and median is calculated.
    '''

    for i in range(4):
        length1=((coords1[2*i+1][0]-Xc1)**2+(coords1[2*i+1][1]-Yc1)**2)**0.5
        length2=((coords1[2*i+2][0]-coords1[2*i+1][0])**2+(coords1[2*i+2][1]-coords1[2*i+1][1])**2)**0.5
        r=length2/length1
        r=r/float(ratios[i])
        if(r>1):
            r=1
        if(r<-1):
            r=-1
        #using current and refernce ratios, finger angles are calculated.
        degs.append(math.degrees(math.acos((r))))
    
    ###Solved Formula for calculating Palm and thumb angle.

    #Approx distance between palm-thumb joint and center of palm is calculated.  
    ax=((coords1[0][0]-coords1[9][0])**2+(coords1[0][1]-coords1[9][1])**2)**0.5
    bx=((coords1[1][0]-coords1[9][0])**2+(coords1[1][1]-coords1[9][1])**2)**0.5
    cx=((coords1[0][0]-coords1[1][0])**2+(coords1[0][1]-coords1[1][1])**2)**0.5
    h=(4*(ax**2)*(bx**2)-(ax**2+bx**2-cx**2)**2)/(4*(bx**2))
    h=max(h,0)**0.5
    h=h/float(ratios[4])
    if(h>1):
        h=1
    if(h<-1):
        h=-1
    #using current and refernce ratios, palm angles are calculated.
    degs.append(math.degrees(math.acos((h))))

    #Approx distance between palm-little finger joint and tip of thumb is calculated.
    ax=((coords[0][0]-coords[4][0])**2+(coords[0][1]-coords[4][1])**2)**0.5
    bx=((coords[0][0]-coords[17][0])**2+(coords[0][1]-coords[17][1])**2)**0.5
    cx=((coords[4][0]-coords[17][0])**2+(coords[4][1]-coords[17][1])**2)**0.5
    h=(ax**2+bx**2-cx**2)/(2*ax*bx)
    h=math.degrees(math.acos((max(h,0)**0.5)))
    h=h/float(ratios[5])
    if(h>1):
        h=1
    if(h<-1):
        h=-1
    #using current and refernce ratios, thumb angles are calculated.    
    degs.append(math.degrees(math.acos((h))))    
    
    #angles for current timestep is updated in m_avg and previous nth timestep entry is deleted.
    for i in range(6):
        m_avg[avg_i][i]=degs[i]

def store_deb(coords, t0):
    #This func. calculates the refernce values
    global ratios
    #Exponential averaging is used to get correct ratios for 15 secs..
    #ratio = Beta*ratio+New_Value*(1-Beta)
    
    #First Fifteen seconds are alloted for adjusting hand in camera and next fifteen seconds to set the refernce ratios. 
    if((time.time()-t0)>15):
        #Get the coordinates of the required joints.
        coords1=[]
        for i in [1,5,8,9,12,13,16,17,20,0]:
            coords1.append(coords[i])
        #Get median of all joints
        Xc1=(coords1[0][0]+coords1[1][0]+coords1[3][0]+coords1[5][0]+coords1[7][0])/5
        Yc1=(coords1[0][1]+coords1[1][1]+coords1[3][1]+coords1[5][1]+coords1[7][1])/5
        
        '''Get the ratios of 4 fingers to calculate finger angles.
           ratio of length of finger to distance between finger-palm joint and median is calculated.
        '''
        for i in range(4):
            length1=((coords1[2*i+1][0]-Xc1)**2+(coords1[2*i+1][1]-Yc1)**2)**0.5
            length2=((coords1[2*i+2][0]-coords1[2*i+1][0])**2+(coords1[2*i+2][1]-coords1[2*i+1][1])**2)**0.5
            r=length2/length1
            ratios[i] = 0.8*ratios[i]+r*0.2
        
        #Approx distance between palm-thumb joint and center of palm is calculated.
        ax=((coords1[0][0]-coords1[9][0])**2+(coords1[0][1]-coords1[9][1])**2)**0.5
        bx=((coords1[1][0]-coords1[9][0])**2+(coords1[1][1]-coords1[9][1])**2)**0.5
        cx=((coords1[0][0]-coords1[1][0])**2+(coords1[0][1]-coords1[1][1])**2)**0.5
        h=(4*(ax**2)*(bx**2)-(ax**2+bx**2-cx**2)**2)/(4*(bx**2))
        h=max(h,0)**0.5
        ratios[4]=ratios[4]*0.8+h*0.2

        #Approx distance between palm-little finger joint and tip of thumb is calculated.
        ax=((coords[0][0]-coords[4][0])**2+(coords[0][1]-coords[4][1])**2)**0.5
        bx=((coords[0][0]-coords[17][0])**2+(coords[0][1]-coords[17][1])**2)**0.5
        cx=((coords[4][0]-coords[17][0])**2+(coords[4][1]-coords[17][1])**2)**0.5
        h=(ax**2+bx**2-cx**2)/(2*ax*bx)
        h=math.degrees(math.acos((max(h,0)**0.5)))
        ratios[5]=ratios[5]*0.8+h*0.2
        
if __name__ == '__main__':
    tf.app.run()
