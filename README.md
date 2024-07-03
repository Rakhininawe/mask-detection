# mask-detection-using-nano-jetson-2gb
# Mask Detection

## Aim and Objective

### Aim

To create a Mask detection system which will detect Human face and then check if mask  is worn or not.


### Demo 


https://github.com/Rakhininawe/mask-detection/assets/147587956/6a9e8cf0-34c9-4f7b-bb75-cac24ffcdda5

### link:- https://youtu.be/kzw6olS1fB4






## Objective

• The main objective of the project is to create a program which can be either run on Jetson nano or any pc with YOLOv5 installed and start detecting using the camera module on the device. • Using appropriate datasets for recognizing and interpreting data using machine learning. • To show on the optical viewfinder of the camera module whether a person is wearing a Mask or not.

## Abstract

• A person’s eyes is classified whether a Mask is worn or not and is detected by the live feed from the system’s camera. • We have completed this project on jetson nano which is a very small computational device. • A lot of research is being conducted in the field of Computer Vision and Machine Learning (ML), where machines are trained to identify various objects from one another. Machine Learning provides various techniques through which various objects can be detected. • One such technique is to use YOLOv5 with Roboflow model, which generates a small size trained model and makes ML integration easier. • It seems like you're referring to the importance of masks, likely in the context of personal protection or public health. Masks have indeed been recognized as crucial tools in preventing the spread of infectious diseases, especially respiratory illnesses like COVID-19. They help to reduce the transmission of respiratory droplets that may contain viruses or bacteria, thereby protecting both the wearer and those around them.

## Introduction

This project is based on a Mask detection model with modifications. We are going to implement this project with Machine Learning and this project can be even run on jetson nano which we have done.• This project can also be used to gather information about who is wearing a Mask and who is not.Masks worn can be classified into various types based on their construction, intended use, and features. Firstly, there are disposable surgical masks, typically made of non-woven materials like polypropylene. These masks are loose-fitting and primarily designed to prevent the wearer from spreading respiratory droplets to others and to protect against large droplets from others. They are commonly used in healthcare settings and by the general public during outbreaks to reduce transmission.

## Literature Review

• Wearing mask helps to reduce the impact of several vision-related issues and challenges, thereby significantly improving daily life and overall well-being. Primarily, the impact of masks on refractive errors primarily explores how different types of masks affect vision correction. Research has focused on understanding how mask-wearing, particularly during events like pandemics, influences individuals with refractive issues such as myopia, hyperopia, and astigmatism. Studies have examined various types of masks including cloth masks, surgical masks, and respirators (like N95 masks), assessing factors such as fit, material, and prolonged use on visual acuity. Key considerations include the potential for masks to cause fogging, which can affect individuals wearing corrective lenses and may impact near-vision tasks, especially among older adults with presbyopia. The literature highlights the need for strategies to mitigate these effects, such as anti-fogging technologies, and suggests further research to better understand the long-term implications of mask use on visual health in different populations.A literature review focusing on masks and their impact on refractive errors delves into how various types of masks affect vision correction. It examines research investigating the influence of masks—ranging from cloth masks to surgical masks and respirators like N95—on individuals with refractive issues such as myopia, hyperopia, and astigmatism. Studies explore factors such as mask fit, material composition, and prolonged wear, considering their effects on visual acuity and comfort. Significant attention is given to issues like fogging, which can disrupt vision for those wearing corrective lenses, particularly affecting tasks requiring near vision. The review underscores the importance of developing strategies to alleviate these challenges, such as anti-fogging technologies and ergonomic mask designs. Moreover, it identifies gaps in current research, suggesting the need for further investigation into the long-term impacts of mask use on visual health across different demographic groups.

## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers everywhere. • NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run multiple neural networks in parallel for applications like image classification, object detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as little as 5 watts. • Hence due to ease of process as well as reduced cost of implementation we have used Jetson nano for model detection and training. • NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated AI applications. All Jetson modules and developer kits are supported by JetPack SDK. • In our model we have used JetPack version 4.6 which is the latest production release and supports all Jetson modules.

## Nano Jetson 2gb

![image](https://github.com/Rakhininawe/mask-detection/assets/147587956/ea83ce43-efeb-491d-91d8-8055af267aa7)

## Proposed System

1] Study basics of machine learning and image recognition.

2]Start with implementation

• Front-end development • Back-end development

3] Testing, analysing and improvising the model. An application using python and Roboflow and its machine learning libraries will be using machine learning to identify whether a person is wearing a Mask or not.

4] Use datasets to interpret the object and suggest whether the person on the camera’s viewfinder is wearing a Mask or not.

## Methodology

The Mask detection system is a program that focuses on implementing real time Mask detection. It is a prototype of a new product that comprises of the main module: Mask detection and then showing on viewfinder whether the person is wearing a Mask or not. 
Mask Detection Module

This Module is divided into two parts:

1] Mask detection

• Ability to detect the location of a person’s face in any input image or frame. The output is the bounding box coordinates on the detected face of a person. • For this task, initially the Dataset library Kaggle was considered. But integrating it was a complex task so then we just downloaded the images from gettyimages.ae and google images and made our own dataset. • This Datasets identifies person’s face in a Bitmap graphic object and returns the bounding box image with annotation of Mask or no Mask present in each image.

2] No-Mask Detection

• Recognition of the Mask and whether Mask is worn or not. • Hence YOLOv5 which is a model library from roboflow for image classification and vision was used. • There are other models as well but YOLOv5 is smaller and generally easier to use in production. Given it is natively implemented in PyTorch (rather than Darknet), modifying the architecture and exporting and deployment to many environments is straightforward. • YOLOv5 was used to train and test our model for whether the Mask was worn or not. We trained it for 149 epochs and achieved an accuracy of approximately 92%.

## Installation

Initial Configuration

sudo apt-get remove --purge libreoffice* sudo apt-get remove --purge thunderbird* Create Swap

udo fallocate -l 10.0G /swapfile1 sudo chmod 600 /swapfile1 sudo mkswap /swapfile1 sudo vim /etc/fstab /swapfile1 swap swap defaults 0 0 Cuda env in bashrc

vim ~/.bashrc

## add this lines

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1 Update & Upgrade

sudo apt-get update sudo apt-get upgrade Install some required Packages

sudo apt install curl curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py sudo python3 get-pip.py sudo apt-get install libopenblas-base libopenmpi-dev

sudo pip3 install pillow Install Torch

curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

#Check Torch, output should be "True" sudo python3 -c "import torch; print(torch.cuda.is_available())" Install Torchvision

git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision cd torchvision/ sudo python3 setup.py install Clone Yolov5

git clone https://github.com/ultralytics/yolov5.git cd yolov5/ sudo pip3 install numpy==1.19.4

#comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1 sudo pip3 install -r requirements.txt Download weights and Test Yolov5 Installation on USB webcam

sudo python3 detect.py sudo python3 detect.py --weights yolov5s.pt --source 0

## Mask Dataset training

We used Google Colab And Roboflow train your model on colab and download the weights and past them into yolov5 folder link of project

colab file given in repo

## Running Glasses Detection Model


source '0' for webcam

!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

## Advantages

• In fields like security and authentication systems, mask detection helps minimize errors by accurately identifying individuals wearing mask. This reduces false positives and improves the overall reliability of facial recognition technology. • Mask detection system shows whether the person in viewfinder of camera module is wearing a Mask or not with good accuracy. • Its ability to improve accuracy in facial recognition, enhance user experience in virtual try-ons, personalize retail recommendations, and contribute to better medical diagnostics. • Complete automation without the need for user input offers several distinct advantages across various domains. Firstly, it enhances operational efficiency by streamlining processes and reducing dependency on human intervention. Tasks that are automated can be executed seamlessly and consistently, eliminating human errors and ensuring reliability in output. This reliability contributes to improved productivity and cost-efficiency as it minimizes the resources required for oversight and correction of errors.

## Application

• Detects a person’s face and then checks whether Mask is worn or not in each image frame or viewfinder using a camera module. • E-commerce platforms and retail stores have increasingly adopted mask detection technologies to enhance safety measures and compliance with health regulations, especially during the COVID-19 pandemic. These technologies utilize various methods such as computer vision, AI algorithms, and thermal imaging to detect whether individuals entering physical stores or appearing on e-commerce platforms are wearing masks correctly.

## Future Scope

Looking ahead, the future scope of masks extends beyond their current role in mitigating infectious disease transmission. Advances in materials science, wearable technology, and health monitoring suggest a broadening range of applications for masks in various sectors.

## Conclusion

The evolution of masks holds promise for addressing a wide array of challenges and opportunities in the future. Beyond their current role in public health and safety during pandemics, masks are poised to become integral tools in personal health monitoring, workplace safety, environmental sensing, and technological innovation. Advancements in materials science, wearable technology, and sensor integration are paving the way for masks that not only protect against airborne pathogens but also monitor vital health metrics and environmental conditions in real-time. These innovations have the potential to revolutionize healthcare delivery by enabling continuous monitoring of physiological parameters and early detection of health issues.

## Reference

1] Roboflow:- https://roboflow.com/

2] Datasets or images used :- https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

Article
https://doi.org/10.1109/JSEN.2021.3061178

https://doi.org/10.1109/LSP.2020.3032277
