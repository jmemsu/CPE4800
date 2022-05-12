'''
CameraCapture shows a live image from the USB camera. Some camera control
and Image Processing Tools are included
Keys:
    ENTER  - Capture Image
    ESC    - exit
    g      - toggle optimized grayscale conversion
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import time
import numpy as np

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib

def main():

    def decode_fourcc(v):
        v = int(v)
        return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

    font = cv.FONT_HERSHEY_DUPLEX #normal size sans-serif font
    color = (0, 255, 0)

    cap = cv.VideoCapture(0) #0 is the device at VideoIndex0
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)  # Known bug: https://github.com/opencv/opencv/pull/5474

    cv.namedWindow("Video")

    convert_rgb = True
    fps = int(cap.get(cv.CAP_PROP_FPS))
    focus = int(min(cap.get(cv.CAP_PROP_FOCUS) * 100, 2**31-1))  # ceil focus to C_LONG as Python3 int can go to +inf

    cv.createTrackbar("FPS", "Video", fps, 30, lambda v: cap.set(cv.CAP_PROP_FPS, v))
    cv.createTrackbar("Focus", "Video", focus, 100, lambda v: cap.set(cv.CAP_PROP_FOCUS, v / 100))

    while True:
        _status, img = cap.read()

        fourcc = decode_fourcc(cap.get(cv.CAP_PROP_FOURCC))

        fps = cap.get(cv.CAP_PROP_FPS)

        if not bool(cap.get(cv.CAP_PROP_CONVERT_RGB)):
            if fourcc == "MJPG":
                img = cv.imdecode(img, cv.IMREAD_GRAYSCALE)
            elif fourcc == "YUYV":
                img = cv.cvtColor(img, cv.COLOR_YUV2GRAY_YUYV)
            else:
                print("unsupported format")
                break

        #cv.putText(img, "Mode: {}".format(fourcc), (15, 40), font, 1.0, color)
        #cv.putText(img, "FPS: {}".format(fps), (15, 80), font, 1.0, color)
        cv.imshow("Video", img)

        k = cv.waitKey(1)

        #Escape Key in ASCII
        if k == 27:
            break
        #Enter Key in ASCII
        elif k == 13:
            cv.imwrite('/home/pi/CPEProj/IMGCapture.jpeg', img)
        elif k == ord('g'):
            convert_rgb = not convert_rgb
            cap.set(cv.CAP_PROP_CONVERT_RGB, 1 if convert_rgb else 0)

    print('Capture Done')
    time.sleep(2)
    imgFruit = cv.imread('IMGCapture.jpeg')
    ymin, ymax = 65, 365
    xmin, xmax = 160, 520
    imgFruit = imgFruit[ymin:ymax, xmin:xmax]
    modelSizePoints = (100, 100)
    imgFruit = cv.resize(imgFruit, modelSizePoints, interpolation= cv.INTER_LINEAR)
    cv.imwrite('/home/pi/CPEProj/IMGCaptureFruit.jpeg', imgFruit)
    
    imgWeight = cv.imread('IMGCapture.jpeg')
    ymin, ymax = 450, 540
    xmin, xmax = 170, 500
    imgWeight = imgWeight[ymin:ymax, xmin:xmax]
    cv.imwrite('/home/pi/CPEProj/IMGCaptureWeight.jpeg', imgWeight)
    print('Crop/Sizing Done')
    time.sleep(2)

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()






def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

time.sleep(2)
def main2():
  #parser = argparse.ArgumentParser(
  #    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  #parser.add_argument(
  #    '--model', help='File path of .tflite file.', required=True)
  #parser.add_argument(
  #    '--labels', help='File path of labels file.', required=True)
  #args = parser.parse_args()

  #labels = load_labels(args.labels)
  labels = load_labels("FruitDetection.txt")

  #interpreter = Interpreter(args.model)
  interpreter = Interpreter("FruitDetection.tflite")
  interpreter.allocate_tensors()
  _, height, width, _ = interpreter.get_input_details()[0]['shape']

  #img = cv.imread('IMGCapture.jpeg')
  img = Image.open('IMGCaptureFruit.jpeg')
  #(left, upper, right, lower) = (300, 186, 524, 410)
  #img_crop = img.crop((left, upper, right, lower))
  #img.show()
  #img_crop.show()
#  width = 224
#  height = 224
#  size = (width, height)
#  img = cv.resize(img, size, interpolation=cv.INTER_AREA)
#  cv.imshow('Image Sized', img)
#  cv.imwrite('/home/pi/CPEProj/IMGCaptureSized.jpeg', img)
#  img = Image.open('IMGCaptureSized.jpeg')
  results = classify_image(interpreter, img)
  label_id, prob = results[0]
  print(results)
  print(labels[label_id])
  label = labels[label_id]
  with open('label.txt','w') as f:
    f.write(label)
  
  #cv_img = np.array(img)
  #cv.putText(cv_img, label, (0, 0), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
  #cv.imshow('TF matched image', cv_img)
  #cv.imwrite('/home/pi/CPEProj/IMGCapLab.jpeg', cv_img)
  print(img)
  img_label = ImageDraw.Draw(img)
  #system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
  #print(system_fonts)
  #'/usr/share/fonts/truetype/lato/Lato-SemiboldItalic.ttf'
  this_font = ImageFont.truetype("Lato-SemiboldItalic.ttf", 25)
  #this_font = ImageFont.load("arial.pil")
  img_label.text((0,0), label, font=this_font, fill=(255, 255, 0))
  img.show()

#if __name__ == '__main__':
main2()

time.sleep(2)

# Read Image
BGR = cv.imread('IMGCaptureWeight.jpeg')
#BGR = cv.rotate(BGR, cv.ROTATE_90_CLOCKWISE)
RGB = cv.cvtColor(BGR, cv.COLOR_BGR2RGB)

# Channels split
R = BGR[...,2]
G = BGR[...,1]
B = BGR[...,0]

# Threshold per channel
R[B>120] = 0
R[G>120] = 0
R[R<230] = 0

gray = cv.cvtColor(RGB, cv.COLOR_RGB2GRAY)

#binaryThresh = input('Binary Threshold: ')
binaryThresh = 120
# Binarize
Binary = cv.threshold(gray, int(binaryThresh), 255, cv.THRESH_BINARY)[1]
# Edge Detection
Edges = cv.Canny(Binary, 50, 200)

foundNums = {'':''}


for num in range(10):
    templatePATH = '/home/pi/CPEProj/SevenSegmentImages/Templates/' + str(num) + '.jpg'
    # Read Template
    templBGR = cv.imread(templatePATH)
    templRGB =  cv.cvtColor(templBGR, cv.COLOR_BGR2RGB)
    templateGray =  cv.cvtColor(templBGR, cv.COLOR_BGR2GRAY)
    # Binarize Template
    templateBinary = cv.threshold(templateGray, 84, 255, cv.THRESH_BINARY)[1]
    # Denoise Template
    templateFiltered = cv.medianBlur(templateBinary,7)
    # Resize Template
    template = cv.resize(templateFiltered, (templBGR.shape[1]//2, templBGR.shape[0]//2))
    # Edge Detection Template
    templateEdges = cv.Canny(template, 50, 200)
    # Extract Dimensions
    h, w = template.shape

    res1 = cv.matchTemplate(  Edges,templateEdges,cv.TM_SQDIFF)
    res2 = cv.matchTemplate(Edges,templateEdges,cv.TM_SQDIFF_NORMED)
    threshold = 0.25
    locations = np.where(res2 >= threshold)
    print(locations)
    res3 = cv.matchTemplate(Edges,templateEdges,cv.TM_CCORR)
    res4 = cv.matchTemplate(Edges,templateEdges,cv.TM_CCORR_NORMED)
    res5 = cv.matchTemplate(Edges,templateEdges,cv.TM_CCOEFF)
    threshold = 0.75
    locations = np.where(res5 >= threshold)
    centroid = locations.mean(axis=0)
    
    foundNums[str(num)]=str(centroid)

    print(locations)
    print(res5)
    res6 = cv.matchTemplate(Edges,templateEdges,cv.TM_CCOEFF_NORMED)

    (_, _, _, maxLoc5) = cv.minMaxLoc(res5)


    img5 = RGB.copy()
    cv.rectangle(img5, (maxLoc5[0], maxLoc5[1]), (maxLoc5[0] + w, maxLoc5[1] + h), (255,255,128), 2)

    plt.subplot(1,4,1)
    plt.imshow(RGB)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.imshow(Binary, cmap='gray')
    plt.title('Segmented')
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.imshow(templateBinary)
    plt.title('Template')
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.imshow(img5)
    plt.title('Result5')
    plt.axis('off')
    
    plt.savefig('temp.jpg', dpi=250)
    imgTemp = Image.open('temp.jpg')
    imgTemp.show()
    imgTemp.close()
    
    #plt.show(block="false")
    #time.sleep(1)
    #plt.close()
    



writeWeight = open(r'weightResult.txt', 'w')
writeWeight.write(str(weight))
writeWeight.close()

time.sleep(4)
import csv as csv
#In other scripts, the label found by the object is stored to a text file
#Here is read and stored in a string
with open('label.txt', 'r') as labelFile:
	label=labelFile.readlines()
	label="".join(label)
	print(label)
spaces = " " in label




#Much the same as the label, here the weight results are stored to a file and read into the program
with open('weightResult.txt', 'r') as weightFile:
	weight=weightFile.readlines()
	weight="".join(weight)
	weight=int(weight)
	print(str(weight) + " grams")
	
#Open the csv file with calories per 100 grams
fruitCalFile = open('FruitFoodDataMin.csv')
csvreader = csv.reader(fruitCalFile)

#Store the headings from the CSV in an array
headings = []
headings = next(csvreader)

#Store the rows from the CSV in an array
rows = []
for row in csvreader:
	rows.append(row)
#Close the csv file
fruitCalFile.close()
#Move the csv rows into a dictionary for searching
fruitDict = {x[0]: x[1:] for x in rows}
#If the label is found the row is returned here
found = fruitDict[label]
print(found)
#Calories per 100 grams of the food is in the second row
x = found[1]
#Calculation for the calories from our detected food
calResult = (float(weight)/100.0) * float(x)
calResult = str(calResult)
print(calResult + " Calories")

imgRes = Image.open('IMGCapture.jpeg')
imgResText = ImageDraw.Draw(imgRes)
this_font = ImageFont.truetype("Lato-SemiboldItalic.ttf", 25)
imgResText.text((0,0), label, font=this_font, fill=(255, 255, 0))
imgResText.text((0,90), str(weight) + " grams", font=this_font, fill=(255, 255, 0))
imgResText.text((0,180), calResult + " Calories", font=this_font, fill=(255, 255, 0))
imgRes.show()
print("End")

