# Timeline - "You're a wizard Harry!"

This project is made for the April 18th Mini-Hackathon. Our idea at first was to make either a computer vision hand gesture sensing AI, or a timeline application for tracking you and your friends memories. In the end, we decided to combine them. We created a script that tracks your hand gestures to navigate around the web app. Once you input dates, events, optionally descriptions and/or photos, you're able to look around the page like magic. If you make a mistake or want to forget a memory you're also able to delete the memory node. 

## Gestures

### Move left and right
To move left and right on the timeline, use your index finger. Point left or right at a slight upwards tilt. If you want to move faster in one direction, add fingers to point in the direction of your choosing (up to four). If you point with 4 fingers it will move the fastest.

### Preview event / Close preview
To preview the event and details, face your palm to the screen with an open hand. To close the preview, close fist and point knuckles to the camera

### Open event / Close event
To open the event, face both hand's palms to the camera. like you're pushing something onto a shelf, with your wrists at a 90 degree angle. To close the event, close fist and point knuckles at camera. 

## Get started

1) Create a python venv with python 3.8 up to 3.12. No higher no lower.
    - FOR WINDOWS
    ```
    (py -3.10 -m venv -nameEnvHere-)
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    .\-nameEnvHere-\Scripts\activate
    ```
    - FOR LINUX
    - FOR MAC
    `brew install python@3.10`

2) intall the requirements.txt file (pip install -r requirements.txt)
3) run hand\_tracker.py file on terminal and open up timeline file from files (python .\hand_tracker.py)
4) point at 30 degree angle with back of hand to camera

