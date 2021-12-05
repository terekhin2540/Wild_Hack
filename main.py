from imageai.Detection import ObjectDetection
import os
import cv2

execution_path = os.getcwd()

print('execution_path: ', execution_path)

dir_name = os.path.join(execution_path, 'фоточки')
our_directory = os.listdir(dir_name)

print('our_directory: ', our_directory)

directory = os.path.join(dir_name ,'animals_good')

print('directory: ', directory)

if os.path.exists(os.path.join(dir_name, 'animals_good')):
    pass
else:
    os.mkdir(os.path.join(dir_name, 'animals_good'))

count = 1

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
detector.loadModel()

for item in our_directory:
    if item.endswith(".jpg") or item.endswith(".jpeg") or item.endswith(".JPG"):
        print(item)
        #img = cv2.imread(os.path.join(dir_name, f'{item}'), 1)
        #print(img)
        print(os.path.join(dir_name, str(item)))
        detections = detector.detectObjectsFromImage(input_image=os.path.join(dir_name, str(item)),   ## сейчас вот тут происходит ошибка, инпут файл пустой пишетЮ,
                                                     output_image_path=os.path.join(dir_name, f'result_{item}'),
                                                     minimum_percentage_probability=40,
                                                     display_object_name=False)

        # print(detections)
        list_values = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'] ## это список животных, которые может распознать модель
        detect_list = []
        for eachObject in detections:
            detect_list.append(eachObject["name"])

        amount_of_cross = list(set(list_values) & set(detect_list))

        print(amount_of_cross)

        if len(amount_of_cross) > 0:
            img = cv2.imread(os.path.join(dir_name, f'{item}'), 1)
            cv2.imwrite(os.path.join(directory, f'{item}'), img)  ##Добавляет в папку следующую, если есть животное

            print('есть пересечение, можно добавлять в следующую папку: ', count)

        count += 1
        print(detect_list)


