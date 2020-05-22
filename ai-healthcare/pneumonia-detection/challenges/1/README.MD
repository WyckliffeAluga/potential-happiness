In this scenario, you have a radiologist who works in a very busy emergency department in a major city. They are often getting hundreds of emergency images that need to be read every day, and there is no prioritization around those images because they come in through the emergency department, so everything is marked as "urgent." In the current setting, radiologists read these images in a first-in-first-out queue, where all images are simply read in the order that they come in. From a clinical perspective, you know that some urgent cases are truly more urgent than others. From your research in interviewing emergency doctors and radiologists, you have identified that two of the most urgent types of findings on an image are a brain bleed and an aortic dissection. Both of these problems can lead to patient death within minutes, but they can only be detected on imaging, so it is critical these images are read ASAP.

You have used deep learning to create two classification algorithms, one for the detection of brain bleeds on head CT images, and one for the detection of aortic dissection on chest x-ray images. Now, you need to figure out how to integrate these algorithms into the radiologist's workflow so that they are most helpful clinically.

In this exercise you'll be given the following:

A list of images that have come in through the ED in order of patient arrival
Probabilities of 'brain bleed' for each image, as determined by one of your deep learning algorithms
Probabilities of 'aortic dissection' for each image, as determined by one of your deep learning algorithms
You will need to do the following:

Calculate the amount of time it will take before each image is read by the radiologist, given the patient arrival queue, assuming each image takes 6 minutes to read
Implement a heuristic that uses the probabilities returned by your two algorithms to re-order the priority read list, assuming that brain bleeds and aortic dissections are equally urgent
Calculate the time delta for each image between the initial and the re-ordered priority reads
