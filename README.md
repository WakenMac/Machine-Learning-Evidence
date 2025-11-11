# HomePiano: A Comparative Study between the Efficiency of Random Forest and Support Vector Machines in a Monocular Augmented Reality Piano Hand-Key Detection Program.

This project is made to build an Augmented Reality (AR) piano using a mix of OpenCV's ArUco and MediaPipe to detect hands and piano on a video feed, as well as Random Forests and Support Vector Machines to predict key presses.

### Features

-   Hand Detection
-   Piano Key Playing
-   Key Press or Hover Differentiation

### Limitations

-   Only supports the top-down view of the piano
-   Requires sufficient lighting for the detection of ArUco markers
-   ArUco markers must not be covered

### File Structure

The code has the following file structure

```{txt}
Machine_Learning_Evidence
├── Machine_Learning_Course
│   ├── Code
│   │   ├── ArUco_Markers
│   │   │   ├── Count_pixel_area
│   │   │   │   └── ArucoArea.py
│   │   │   ├── Creating_an_aruco_marker
│   │   │   │   └── MakingAnArucoMarker.py
│   │   │   └── Detecting_an_aruco_marker
│   │   │       ├── DetectAruco.py
│   │   │       └── DetectArucoLive.py
│   │   ├── Dataset_Generation
│   │   │       ├── GenerateDataset.py
│   │   │       ├── QueuedList.py
│   │   │       └── VideoPlayer.py
│   │   |
│   │   └── Hand_Detection
│   │       ├── DroidCam.py
│   │       ├── ImageReader.py
│   │       ├── MediaPipePractice.py
│   │       └── temp_coords.py
│   │       
│   ├── Images
│   │   ├── ArUco_Markers
│   │   │   ├── 100px
│   │   │   ├── 200px
│   │   │   ├── 250px
│   │   │   ├── 300px
│   │   │   ├── Detect
│   │   │   └── Digital Piano
│   │   ├── Annotated_Images
│   │   └── Normal_Images
│   └── Trial Recordings
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

\
To get started, make sure you install all required dependencies.

## 🚀 Setup Instructions

We will be using Python 3.10.0 Before running any scripts, install the dependencies by executing:

``` python
pip install -r requirements.txt
```
