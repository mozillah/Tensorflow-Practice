import sys
import os
import numpy as np
import cv2
import time
import copy
from datetime import datetime
from collections import OrderedDict
import argparse
from lxml import etree
import xml.etree.cElementTree as ET
from os.path import join
from glob import glob



def main(video_name="",output_video=""):
    print("video")
    def nonblank_lines(f):
        for l in f:
            line = l.rstrip()
            if line:
                yield line


    def write_xml(xml_str, xml_path):
        # remove blank text before prettifying the xml
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.fromstring(xml_str, parser)
        # prettify
        xml_str = etree.tostring(root, pretty_print=True)
        # save to file
        with open(xml_path, 'wb') as temp_xml:
            temp_xml.write(xml_str)


    def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
        # By: Jatin Kumar Mandav
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = folder_name
        ET.SubElement(annotation, 'filename').text = image_name
        ET.SubElement(annotation, 'path').text = abs_path
        source = ET.SubElement(annotation, 'source')
        ET.SubElement(source, 'database').text = 'Unknown'
        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = img_width
        ET.SubElement(size, 'height').text = img_height
        ET.SubElement(size, 'depth').text = depth
        ET.SubElement(annotation, 'segmented').text = '0'

        xml_str = ET.tostring(annotation)
        write_xml(xml_str, xml_path)

    def voc_format(class_name, point_1, point_2):
        # Order: class_name xmin ymin xmax ymax
        xmin, ymin = min(point_1[0], point_2[0]), min(point_1[1], point_2[1])
        xmax, ymax = max(point_1[0], point_2[0]), max(point_1[1], point_2[1])
        items = map(str, [class_name, xmin, ymin, xmax, ymax])

        return items
        
    def append_bb(ann_path, line, extension):
        if '.txt' in extension:
            with open(ann_path, 'a') as myfile:
                myfile.write(line + '\n') # append line
        if '.xml' in extension:

            class_name, xmin, ymin, xmax, ymax = line

            tree = ET.parse(ann_path)
            annotation = tree.getroot()

            obj = ET.SubElement(annotation, 'object')
            ET.SubElement(obj, 'name').text = class_name
            ET.SubElement(obj, 'pose').text = 'Unspecified'
            ET.SubElement(obj, 'truncated').text = '0'
            ET.SubElement(obj, 'difficult').text = '0'

            bbox = ET.SubElement(obj, 'bndbox')
            ET.SubElement(bbox, 'xmin').text = xmin
            ET.SubElement(bbox, 'ymin').text = ymin
            ET.SubElement(bbox, 'xmax').text = xmax
            ET.SubElement(bbox, 'ymax').text = ymax

            xml_str = ET.tostring(annotation)
            write_xml(xml_str, ann_path)


    def get_annotation_paths(img_path, annotation_formats):
        annotation_paths = []
        for ann_dir, ann_ext in annotation_formats.items():
            new_path = os.path.join(OUTPUT_DIR, ann_dir)
            new_path = img_path.replace(INPUT_DIR, new_path, 1)
            pre_path, img_ext = os.path.splitext(new_path)
            new_path = new_path.replace(img_ext, ann_ext, 1)
            annotation_paths.append(new_path)
        return annotation_paths

    def save_bounding_box(annotation_paths, class_index, point_1, point_2, width, height):
        for ann_path in annotation_paths:
            if '.txt' in ann_path:
                line = yolo_format(class_index, point_1, point_2, width, height)
                append_bb(ann_path, line, '.txt')
            
            print(ann_path)
            if '.xml' in ann_path:
                line = voc_format(class_[class_index], point_1, point_2)
                append_bb(ann_path, line, '.xml')


    def yolo_format(class_index, point_1, point_2, width, height):
        # YOLO wants everything normalized
        # Order: class x_center y_center x_width y_height
        x_center = (point_1[0] + point_2[0]) / float(2.0 * width)
        y_center = (point_1[1] + point_2[1]) / float(2.0 * height)
        x_width = float(abs(point_2[0] - point_1[0])) / width
        y_height = float(abs(point_2[1] - point_1[1])) / height
        items = map(str, [class_index, x_center, y_center, x_width, y_height])
        return ' '.join(items)


    # Get the names of the output layers
    def getOutputsNames(net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        # if DEBUG:
        #     cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 1)
         # Draw a bounding box.
   

        label = '%.2f' % conf
        colorId = (0, 0, 255)

        print(classId)
        # Get the label for the class name and its confidence
        if class_:
            assert(classId < len(class_))
            label = '%s:%s' % (class_[classId], label)
            if classId == 0:
                colorId = (0,255,0)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (left, top), (right, bottom), colorId, 3)
        top = max(top, labelSize[1])

        cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255,255,255), cv2.FILLED)
        # print(class_[classId])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    def postprocess(frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        rects = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        
        classIds = []
        confidences = []
        boxes = []
        star = time.time()
        lis = []
        check = []

        # [check.append(outs[count][n1,:]) for out in outs for detection in out ]
        [check.append(outs[count][n1,:]) for count,out in enumerate(outs) for n1 in np.where(outs[count][:,5:] != 0)[0]]
        

        for detection in check:
            scores = detection[5:]
            classId = np.argmax(scores)
            # if classId > 0:
            #     continue
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

        endr = time.time()
        print("enumerate :   {:.2f}ms".format((endr - star)* 1000))
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        fClasses =[]
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            # if top < 30 : #no detection above 30
            #     continue
            bos = np.array([left,top,width,height])
            rects.append(bos.astype("int"))
            fClasses.append(classIds[i])
            # box = np.array([left,top,left+width,top+height])
            # rects.append(box.astype("int"))
            drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        return rects,fClasses

    DEBUG = True
    starting_time = time.time()

    # video_name= '/home/facit/Desktop/Videos/08-07-2018-13-50.avi'
    # video_name= '/home/facit/Desktop/Videos/2017-05-29-14-/2017-05-29-14-05-00.avi'
    # video_name= 'rtsp://admin:pakistan1947@192.168.2.61:554/profile3/media.smp'
    file_path, file_extension = os.path.splitext(video_name)

    file_extension = file_extension.replace('.', '_')
    file_path += file_extension
    video_name_ext = os.path.basename(file_path)
    if not os.path.exists(file_path):
        print(' Converting video to individual frames...')
        os.makedirs(file_path)


    confThreshold = 0.1  #Confidence threshold
    nmsThreshold = 0.1   #Non-maximum suppression threshold

    inpWidth = inpHeight = 416 #Height of network's input imageai #Width of network's input image
    OUTPUT_DIR ="output"
    INPUT_DIR = "input"
    classesFile = "/home/facit/Desktop/Models/yolo.names"
    classes = None
    modelConfiguration = "/home/facit/Desktop/WeightsFilesBackup/allModel.cfg"
    modelWeights = "/home/facit/Desktop/WeightsFilesBackup/PeopleBest.weights"

    annotation_formats = {'PASCAL_VOC' : '.xml', 'YOLO_darknet' : '.txt'}
    annotation_formats = {'PASCAL_VOC' : '.xml'}

    with open(classesFile, 'rt') as f:
        class_ = f.read().rstrip('\n').split('\n')
    print(class_)
    

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    execution_path = os.getcwd()
    cap = cv2.VideoCapture(video_name)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    video_last_name = video_name.split("\\")
    fps = cap.get(cv2.CAP_PROP_FPS)
    codec=cap.get(cv2.CAP_PROP_FOURCC)
    out = cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc('M','J','P','G'), fps,(frame_width, frame_height))

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    ret, frame = cap.read()
        # Select ROI
    cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)

    frameHeight = frame_height
    frameWidth = frame_width
    i=0
    desired_img_format = '.jpg'
    inc = fps
    start_frame_number=0
    # cv2.setUseOptimized(False)
    while(cap.isOpened()):
    # Capture frame-by-frame
        
        ret, frame = cap.read()
        
        rects = []
        classes=[]
        
        if ret == True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
            start_frame_number = start_frame_number +inc

            # frame = cv2.resize(frame,(int(inpWidth),int(inpHeight)))
            start_time = time.time()
            blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)
            # Sets the input to the network
            net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))
            frame_name =  '{}_{}{}'.format(video_name_ext, i, desired_img_format)
            frame_path = os.path.join(file_path, frame_name)
            cv2.imwrite(frame_path, frame)
            # Remove the bounding boxes with low confidence
            rects,classes =postprocess(frame, outs)
            i+=1
            annotation_paths = get_annotation_paths(frame_path, annotation_formats)
                        
            abs_path = os.path.abspath(frame_path)
            folder_name = os.path.dirname(frame_path)
            image_name = os.path.basename(frame_path)
            img_height, img_width, depth = (str(number) for number in frame.shape)

            if len(classes)>0:
                for ann_path in annotation_paths:
                    if not os.path.isfile(ann_path):
                        create_PASCAL_VOC_xml(ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)


            for box, classId in zip(rects, classes):
                save_bounding_box(annotation_paths, classId, (int(box[0]), int(box[1])),(int(box[0]) + int(box[2]), int(box[1]) + int(box[3])), frameWidth, frameHeight)

            cv2.circle(frame, (0,0) , 4,(255,255,255), -1)
            end_time = time.time()
            # cv2.putText(frame, '{:.2f}ms'.format((end_time - start_time) * 1000), (40, 40), 0,fontScale=1, color=(0, 255, 0), thickness=2)
            print("FPS :   {:.2f}ms".format((end_time - start_time)* 1000))
            cv2.imshow("frame",frame)
            out.write(frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Break the loop
        else: 
            break


    # When everything done, release the video capture object
    cap.release()
    # out.release() 
    cv2.destroyAllWindows()

    ending_time = time.time()
    print("FPS end :   {:.2f}ms".format((ending_time - starting_time)* 1000))



# if __name__ == "__main__":
    



#     location="/home/facit/Downloads/Fertig/Fertig/"
#     folder=location+"*.mp4"
#     # x-special/nautilus-clipboard
#     # file:///home/facit/Desktop/PythonLabelImage/newVideo/Mazz%20Check%20this%20video
#     # file = "/home/facit/Desktop/PythonLabelImage/newVideo/video"
#     # main(video_name=file)
 
#     print(folder)
#     # videofiles =[]
#     files = glob.glob(folder)
#     for file in files:
#         if os.path.isfile(file):    
#             # videofiles.append(file)
#             print(file)
#             main(video_name=file)







if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
    parser.add_argument('--image', help='Path to image file.')
    parser.add_argument('--video', help='Path to video file.')
    parser.add_argument('--folder', help='Path to video file.')
    fileTypes =('*.avi', '*.mp4','*.webm')
    args = parser.parse_args()
    if (args.video):
    # Open the video file
        if not os.path.isfile(args.video):
            print("Input video file ", args.video, " doesn't exist")
            sys.exit(1)
        video_name = args.video
        outputFile = video_name[:-4]+'resthead_output.avi'
        main(video_name=video_name,output_video= outputFile)


    if (args.folder):
        if os.path.exists(args.folder):    
            files = []
            output_folder= join(args.folder, 'new')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            for ext in fileTypes:
#                 import os
                files.extend(glob(join(args.folder, ext)))
            for filepath in files:
                print(filepath)  
                if os.path.isfile(filepath):
                    output_file = os.path.basename(filepath)
                    # outputFile = video_name[:-4]+'_output.avi'
                    filename = os.path.basename(filepath)
                    outfile  = filename[:-4]+'_output.avi'
                    output_file = join(output_folder, outfile)
                    print("Input video file ",filepath)
                    print(output_file)
                    main(video_name=filepath,output_video=output_file)
