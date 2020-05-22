Types of clinical images 

x-ray

The most common type of 2D imaging is x-ray. This technique uses a machine to emit x-rays, which are absorbed differently by different tissues in the body. Bone has high absorption and therefore appears bright white. Soft tissues like the heart and diaphragm absorb a medium amount and appear gray. Air does not absorb any x-rays and thus appears black.

We usually think of x-rays to look for fractures/broken bones, but two of their other most common use cases are for assessing abnormalities in the lungs, and for assessing breast tissue (mammograms).

Computed Tomography (CT)

CT uses x-ray, but they emit x-rays from many different angles around the human body to capture more detail from more different angles. They are more expensive, but they allow us to see more details about organs and soft tissues in the body. Most hospitals in the US have a CT scanner in them.

Magnetic Resonance Imaging (MRI)

MRI uses strong magnetic fields and radio waves to create images of areas of the body from all different angles. It allows us to assess even more details about the human body. This type of imaging is particularly useful for studying the human brain. Although it is safer in that it does not emit radiation, MRI is not safe for people with metal in their bodies. Not all hospitals in the US have MRI scanners and it is a very expensive imaging tool.

Ultrasound

Ultrasound is a type of 2D imaging technique that isn't covered in the video. It utilizes high-frequency sound waves beyond the audible limit of human hearing to generate images. Ultrasound waves travel through soft tissues or fluids and bounce back when it hits dense tissues. More waves bounce back if the tissue is denser. The waves that bounce back are captured to generate images. Ultrasound is very safe and commonly used during pregnancy.

Microscopy

Microscopy refers to physical slides of biological material taken from a patient that can be viewed at the cell-level through a microscope. These slides often have a stain applied to them that causes different cell structures to appear in different colors. These stains help pathologists tell the difference between cell structures.

Fundal Imaging

The fundus of the eye is the interior surface of the eye, and images can be taken of it to diagnose diabetic retinopathy. In this condition, blood vessels at the back of the eye become damaged, so fundal imaging particularly looks at the integrity of the tiny vessels in the eye.

Differences in the imaging techniques
Since fundal images and microscopy images are not acquired with a digital machine, they are not inherently digital like x-rays are. As a result, an additional step of digitizing these images is required before applying AI. Once microscopy and fundal images are digitized, much of the AI principals can be applied to them the same way that they can be applied to x-ray images.

The second difference is that X-ray images are stored as single-channel grayscale images, while microscopy and fundal images are stored as red-green-blue (RGB) three-channel images.

Another major difference is that x-rays are stored in the DICOM format, which is the standard file format for medical imaging data, while this does not apply to microscopy and fundal images. We will cover more on this later.

2D vs. 3D

Out of these different imaging tools, x-ray is the only 2D imaging tool.

2D imaging takes the picture from a single angle, and everything that is visible to the device at that angle will appear in the picture. You do not see overlapping structures in the 2D image.

3D imaging takes lots of pictures from lots of angles to create a volume of images. You can see structures that are behind one another. The final 3D image is actually a set of 2D images that represent different slices through the body. So, in any single slice or 2D image, you can’t see the whole part but if you scroll through the slices, you will view the volume of that body part.


Clinicians Involved in Medical Imaging


Radiologist

The primary reader of medical imaging data is a type of clinician called a radiologist. These clinicians read all types of 2D and 3D images from all areas of the body. Their role in the clinical workflow is to read imaging studies and write interpretations of the images that can then be understood by other clinicians who are not experts in imaging.

Diagnosing Clinician

After the radiologist reads an imaging study, their radiology report is sent to the patient's diagnosing clinician. This clinician could be an emergency room doctor, primary care physician (PCP), or any other type of specialist. While the radiologist's report may have diagnostic information in it, the final diagnosis always comes from the diagnosing clinician who takes the radiologist's report into account alongside other information: the patient's medical history, lab results, and current symptoms. So, medical imaging plays a critical, but only a partial role in the diagnostic process.

Pathologists

Pathologists are a type of clinician who work primarily in laboratories. While radiologists are the primary readers of x-rays, CT, and MRI studies, pathologists are the primary readers of microscopy studies. It is their job to interpret findings from all different types of cell-level samples taken from patients such as tumor biopsies and blood smears.
