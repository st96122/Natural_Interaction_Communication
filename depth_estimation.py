import os
import re
import random
import json
from glob import glob
import time
import cv2
import numpy as np


# MAIN PATH
ROOT_PATH = os.getcwd()
USER_PATH = ROOT_PATH.split('PycharmProjects')[0]
DATASET_PATH = os.path.join(USER_PATH, 'dataset')

# IMGAES PATH
OPENPOSE_TEST_PATH = os.path.join(ROOT_PATH, 'openpose_test')
IMAGES_PATH = os.path.join(OPENPOSE_TEST_PATH, 'images_org')
ANNOTATIONS_PATH = os.path.join(OPENPOSE_TEST_PATH, 'images_json')
VIDEO_PATH = os.path.join(OPENPOSE_TEST_PATH, 'video_org')
OUTPUT_IMAGES = os.path.join(ROOT_PATH, 'output_images')

# VIDEO PATH
OPENPOSE_TEST_VIDEO_PATH = os.path.join(DATASET_PATH, 'openpose_test_video')
LEE_PATH = os.path.join(OPENPOSE_TEST_VIDEO_PATH, 'lee')
LIN_PATH = os.path.join(OPENPOSE_TEST_VIDEO_PATH, 'lin')
LIU_PATH = os.path.join(OPENPOSE_TEST_VIDEO_PATH, 'liu')
ZHANG_PATH = os.path.join(OPENPOSE_TEST_VIDEO_PATH, 'zhang')
LIU_54CM_PATH = os.path.join(OPENPOSE_TEST_VIDEO_PATH, 'liu_54cm')



def handraise(left_hand,right_hand):
    """
               右肩（0,1) 0
               右肘 (2,3) 1
               右腕 (4,5) 2

               左肩（6,7) 0
               左肘 (8,9) 1
               左腕 (10,11) 2
               """
    l=0
    r=0

    if ((right_hand[0,0] - right_hand[2,0]) != 0):

        rm1 = (right_hand[0,1] - right_hand[2,1]) / (right_hand[0,0] - right_hand[2,0])  # >0 舉右手
        if rm1 > 0 and right_hand[2,1] < right_hand[0,1]:
            r=1

    if ((left_hand[0,0] - left_hand[2,0]) != 0):
        lm1 = (left_hand[0,1] - left_hand[2,1]) / (left_hand[0,0] - left_hand[2,0])  # >0 放右手
        if lm1 < 0 and left_hand[2,1] <left_hand[0,1]:
            l=1
    return l,r

def load_signal_data():
    """
    load data from preprocessing images, annotations and return the abspath back

    :return: left_images_path, right_images_path, left_annotations_path, right_annotations_path
    """
    # Load images, annotations path
    images = sorted(glob(IMAGES_PATH + '/*'))
    annotations = sorted(glob(ANNOTATIONS_PATH + '/*'))

    patten = 'left'
    left_images, right_image = [], []
    left_annotations, right_annotations = [], []
    for image, annotation in zip(images, annotations):
        if len(re.findall(patten, image)):
            left_images.append(image)
            left_annotations.append(annotation)
        else:
            right_image.append(image)
            right_annotations.append(annotation)
    return left_images, right_image, left_annotations, right_annotations


def load_stream_data(test_path, test_file=0):
    """
    load data from preprocessing video, annotations and return the abspath back

    :return: left_video, right_video, left_annotations_path, right_annotations_path
    """
    left_video_file = os.path.join(test_path + '/left_{:02d}.avi'.format(test_file))
    right_video_file = os.path.join(test_path + '/right_{:02d}.avi'.format(test_file))
    left_annotations_dir = os.path.join(test_path + '/left_{:02d}_json/*'.format(test_file))
    right_annotations_dir = os.path.join(test_path + '/right_{:02d}_json/*'.format(test_file))

    left_annotations = sorted(glob(left_annotations_dir))
    right_annotations = sorted(glob(right_annotations_dir))

    return left_video_file, right_video_file, left_annotations, right_annotations


def load_body_points(left_json, right_json):
    """
    open the json file and pick the best accuracy body port return.

    :param left_json: left image best body port
    :param right_json: right image best body port
    :return:
    """
    left_hand=np.zeros((3,2),dtype=np.float)
    right_hand = np.zeros((3, 2), dtype=np.float)
    left_body = [2, 3, 4, 8, 9, 10]
    right_body = [5, 6, 7, 11, 12, 13]
    left_face = [14, 16]
    right_face = [15, 17]
    with open(left_json, 'r') as f:
        left_annotation = json.load(f)
        # find the largest confidence people
        peoples = left_annotation['people']
        confidence = 0
        left_points = np.array([])
        for people in peoples:
            keep = np.array(people['pose_keypoints_2d']).reshape((-1, 3))
            if np.mean(keep[:, 2]) > confidence:
                confidence = np.mean(keep[:, 2])
                left_points = np.copy(keep)
        # left_points = np.array(left_annotation['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
        # body left right points accuracy take (left + right / 2)
        k=0
        for i in [2,3,4]:
            for j in [0,1]:
                right_hand[k,j]=left_points[i,j]
            k+=1
        k=0
        for i in [5,6,7]:
            for j in [0,1]:
                left_hand[k,j]=left_points[i,j]
            k+=1

        left_points[left_body, 2], left_points[right_body, 2] = [(left_points[left_body, 2] + left_points[right_body, 2])
                                                                 / 2 for i in [0, 1]]
        left_points[left_face, 2], left_points[right_face, 2] = [(left_points[left_face, 2] + left_points[right_face, 2])
                                                                 / 2 for i in [0, 1]]

    with open(right_json, 'r') as f:
        right_annotation = json.load(f)
        # find the largest confidence people
        peoples = right_annotation['people']
        confidence = 0
        right_points = np.array([])
        for people in peoples:
            keep = np.array(people['pose_keypoints_2d']).reshape((-1, 3))
            if np.mean(keep[:, 2]) >= confidence:
                confidence = np.mean(keep[:, 2])
                right_points = np.copy(keep)
        #right_points = np.array(right_annotation['people'][0]['pose_keypoints_2d']).reshape((-1, 3))
        # body left right points accuracy take (left + right / 2)
        right_points[left_body, 2], right_points[right_body, 2] = [(right_points[left_body, 2] + right_points[right_body, 2])
                                                                   / 2 for i in [0, 1]]
        right_points[left_face, 2], right_points[right_face, 2] = [(right_points[left_face, 2] + right_points[right_face, 2])
                                                                   / 2 for i in [0, 1]]

    index = np.argmax((left_points[:, 2] + right_points[:, 2]))
    # it is possible cause left right up site down
    if index in left_body:
        return (left_points[index]+left_points[index+3])/2, (right_points[index]+right_points[index+3])/2,left_hand,right_hand
    elif index in right_body:
        return (left_points[index]+left_points[index-3])/2, (right_points[index]+right_points[index-3])/2,left_hand,right_hand
    elif index in left_face:
        return (left_points[index]+left_points[index+1])/2, (right_points[index]+right_points[index+1])/2,left_hand,right_hand
    elif index in right_face:
        return (left_points[index]+left_points[index-1])/2, (right_points[index]+right_points[index-1])/2,left_hand,right_hand

    return left_points[index], right_points[index],left_hand,right_hand


# Load images, annotations data
def depth_estimation(x_left, x_right, f=33.4, d=114):
    """
    Calculation the people depth
    :param x_left: left image x point
    :param x_right: right image x point
    :param f: focal length
    :param d: two camera distance
    :return:
    """
    depth = abs(f * d / ((x_left - x_right) / 72 * 2.54)) / 100  - 0.418879
    dg = 90 - ((x_left - 1280 / 2) / 1280) * 78
    dg2 = 90 - (((1280 / 2)-x_right) / 1280) * 78
    depth2 = abs(110 * np.sin(dg * np.pi / 180) * np.sin(dg2 * np.pi / 180) / np.sin((dg + dg2) * np.pi / 180)) / 100
    return depth,depth2


def accuracy(ground_truth, depth):
    if type(ground_truth) == str:
        ground_truth = int(ground_truth.split('_')[1].split('m')[0])
    return 100 - abs(depth - ground_truth) / ground_truth * 100


def test_image(index=None):
    left_images, right_images, left_annotations, right_annotations = load_signal_data()
    if index is None:
        index = random.randint(0, len(left_images) - 1)
    filename = os.path.split(left_images[index])[-1].split('.')[0]
    try:
        left_point, right_point = load_body_points(left_annotations[index], right_annotations[index])
        depth = depth_estimation(left_point[0], right_point[0])
        acc = accuracy(filename, depth)
    except IndexError:
        left_point, right_point = [0, 0, 0], [0, 0, 0]
        depth = 0
        acc = 0
    # Image show
    l_img = cv2.imread(left_images[index])
    r_img = cv2.imread(right_images[index])
    cv2.circle(l_img, (int(left_point[0]), int(left_point[1])), 3, (0, 255, 0), -1)
    cv2.circle(r_img, (int(right_point[0]), int(right_point[1])), 3, (0, 255, 0), -1)

    output_img = np.hstack([r_img, l_img])
    text1 = 'Point accuracy: {:.2f}%'.format((left_point[2] + right_point[2])/2)
    text2 = 'Depth: {:.2f}m'.format(depth)
    text3 = 'Accuracy: {:.2f}%'.format(acc)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output_img, text1, (100, 100), font, fontScale=1,
                color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(output_img, text2, (100, 150), font, fontScale=1,
                color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(output_img, text3, (100, 200), font, fontScale=1,
                color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow(filename, output_img)
    # cv2.imwrite(os.path.join(OUTPUT_IMAGES, filename + '.png'), output_img)
    key = cv2.waitKey()
    cv2.destroyAllWindows()
    return key


def test_images():
    left_images, right_images, left_annotations, right_annotations = load_signal_data()

    acc_5m, acc_10m, acc_15m, acc_20m = [[] for i in range(4)]
    for left_annotation, right_annotation in zip(left_annotations, right_annotations):
        filename = os.path.split(left_annotation)[-1].split('.')[0]
        try:
            left_point, right_point,left_hand,right_hand = load_body_points(left_annotation, right_annotation)
            depth = depth_estimation(left_point[0], right_point[0])
            acc = accuracy(filename, depth)
        except IndexError:
            acc = 0
        if filename.split('_')[1] == '5m':
            acc_5m.append(acc)
        elif filename.split('_')[1] == '10m':
            acc_10m.append(acc)
        elif filename.split('_')[1] == '15m':
            acc_15m.append(acc)
        elif filename.split('_')[1] == '20m':
            acc_20m.append(acc)

    print("5m Accuracy: {:.2f}%".format(np.mean(acc_5m)))
    print(acc_5m)
    print("10m Accuracy: {:.2f}%".format(np.mean(acc_10m)))
    print(acc_10m)
    print("15m Accuracy: {:.2f}%".format(np.mean(acc_15m)))
    print(acc_15m)
    print("20m Accuracy: {:.2f}%".format(np.mean(acc_20m)))
    print(acc_20m)


def test_video(mode='video', test_path='/', test_file=0, ground_truth=None):
    WIDTH = 1280
    HEIGHT = 720
    if mode == 'camera':
        # Video from camera
        cap1 = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1)
        cap1.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        cap2.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        # TODO: need to connect openpose output
        left_annotations = []
        right_annotations = []
    else:
        # Video from file
        video_lift, video_fight, left_annotations, right_annotations = load_stream_data(test_path, test_file)
        cap1 = cv2.VideoCapture(video_lift)
        cap2 = cv2.VideoCapture(video_fight)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = os.path.join(test_path, 'depth_output_video/')
    output_filename = 'output_video_{:02d}.avi'.format(test_file)
    out = cv2.VideoWriter(output_video + output_filename, fourcc, 20.0, (WIDTH * 2, HEIGHT))
    index = 0
    while cap1.isOpened() and cap2.isOpened():
        # Capture frame-by-frame
        _, l_img = cap1.read()
        _, r_img = cap2.read()
        if l_img is None or r_img is None:
            break

        try:
            left_point, right_point,left_hand,right_hand = load_body_points(left_annotations[index], right_annotations[index])
            # TODO: add
            l_raise,r_raise=handraise(left_hand,right_hand)

            depth,depth2 = depth_estimation(left_point[0], right_point[0], d=110)
            if ground_truth:
                acc = accuracy(ground_truth, depth)
                acc2=accuracy(ground_truth,depth2)
        except IndexError:
            left_point, right_point = [0, 0, 0], [0, 0, 0]
            depth = 0
            depth2=0
            acc = 0
            acc2=0

        # draw circle at detect point
        cv2.circle(l_img, (int(left_point[0]), int(left_point[1])), 4, (0, 255, 0), -1)
        cv2.circle(r_img, (int(right_point[0]), int(right_point[1])), 4, (0, 255, 0), -1)

        # Show Video
        output_img = np.hstack([l_img, r_img])
        text1 = 'Point accuracy: {:.2f}%'.format((left_point[2] + right_point[2]) / 2)
        text2 = 'Depth: {:.2f}m'.format(depth)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(output_img, text1, (100, 100), font, fontScale=1,
                    color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.putText(output_img, text2+'      d2    '+str(depth2), (100, 150), font, fontScale=1,
                    color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        if ground_truth:
            text3 = 'Accuracy1: {:.2f}  Accuracy2: {:.2f}'.format(acc, acc2)
            cv2.putText(output_img, text3, (100, 200), font, fontScale=1,
                        color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        if l_raise==1:
            cv2.putText(output_img,'left',(100,250),font,fontScale=1,color=(0,0,0),thickness=2,lineType=cv2.LINE_AA)
        if r_raise == 1:
            cv2.putText(output_img, 'right', (100, 250), font, fontScale=1, color=(0, 0, 0), thickness=2,
                        lineType=cv2.LINE_AA)
        cv2.imshow('Video', output_img)

        #time.sleep(0.03)
        # Save Video
        out.write(output_img)

        # change to next annotation file
        index += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap1.release()
    cap2.release()
    out.release()
    cv2.destroyAllWindows()


def images_disparity(l_img, r_img):
    l_img = cv2.imread(l_img)
    r_img = cv2.imread(r_img)
    # TODO: add two images together


if __name__ == '__main__':
    # test image pipeline
    # test_image(1)
    # for i in range(24):
    #     key = test_image(i)
    #     if key == 27:
    #         break

    # test images pipeline
    # test_images()

    # test video pipeline
    test_video(mode='video', test_path=LIN_PATH, test_file=3, ground_truth=15)

