from imageai.Detection import ObjectDetection
import os
import cv2

execution_path = os.getcwd()
print(execution_path)
if os.path.exists('images_with_animals'):
    pass
else:
    os.mkdir('images_with_animals')

directory = execution_path+'\images_with_animals'

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path , "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "q.JPG"), output_image_path=os.path.join(execution_path , "q_res.JPG"), minimum_percentage_probability=70)

#print(detections)
list_values = ['person']
detect_list = []
for eachObject in detections:
    #print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    detect_list.append(eachObject["name"])

amount_of_cross = list(set(list_values) & set(detect_list))

if len(amount_of_cross) > 0:
    #os.path.join(f'{execution_path}\images_with_animals', "people_res.jpg")
    #os.path.join('D:/PYTHON/Hackaton/images_with_animals', "people_res.jpg")
    #cv2.imwrite('D:/PYTHON/Hackaton/images_with_animals', "people_res.jpg")

    os.path.join(directory, "q.JPG")

    print('есть пересечение, можно добавлять в следующую папку')
    #cv2.imwrite(os.path.join(directory , 'people_res.jpg'), 'people_res.jpg')

print(detect_list)