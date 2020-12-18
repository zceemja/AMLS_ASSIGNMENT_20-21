# UCL ELEC0134 Assignment 
## Applied Machine Learning Systems

## Key porints

* Run project with **main.py**
* Before running project check and adjust **settings.py**, you can specify _shape_predictor_68_face_landmarks.dat_ location if you have one, otherwise it will be downloaded. 
* In image grid window, any key will show next image batch, use `ESC` to exit.
## Project structure

### Directories
 * **A1, A2, B1, B2** - Each task is structured as python modules in their corresponding directories.
 * **Common** - Python module that contains code shared between task.
 * **Data** - Generated, cached or downloaded data. Can be changed in **settings.py**.
 * **Datasets** - All training and testing datasets.
 * **report** - Latex report source files.

### File Roles
 * **main.py** - top level script that run each task
 * **settings.py** - contains all necessary settings for project to run, including dataset directory locations and other customisations.
 * **Common/common.py** - contains parent model class that is used in each task. Also setups logging.
 * **Common/face_detect.py** - all logic to detect faces, facial landmarks, and download necessary dependencies for them.
 * **Common/ui.py** - helper functions to build UI for graphs and images.
 * **Common/utils.py** - any other helper functions used throughout project.
 
## Required Packages

Note that this project uses python 3.9, however it should work with >=3.5 because of typing support.

Used libraries:

* tensorflow 2.4.0-rc4
* numpy 1.19.4
* dlib 19.21.1
* cv2 4.5.0
* sklearn 0.23.2
* matplotlib 3.3.3 (only needed if `SHOW_GRAPHS=True`)
* requests 2.25.0 (only needed to download missing models)