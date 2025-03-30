# <img src="https://raw.githubusercontent.com/isaiahnields/vidcycle/master/logo.png" width="48"> VidCycle: Enhance Your Cycling Videos with GPS Data Overlay

Welcome to VidCycle, the innovative Python program designed for cycling enthusiasts and professionals alike. VidCycle transforms your cycling experiences by seamlessly integrating Garmin GPS bike computer data with your video footage. The tool overlays vital cycling metrics such as speed, elevation, distance, and heart rate onto your videos, creating an immersive and informative visual experience. 

With VidCycle, you can relive your rides with a modern, clean overlay that enriches your video content without overpowering it. Whether you're analyzing your performance, sharing your adventures with friends, or creating content for your audience, VidCycle offers a unique way to showcase your cycling journeys.

### Key Features:
- **GPS Data Integration**: Automatically syncs with Garmin GPS bike computer data.
- **Customizable Overlays**: Choose what data to display and how it appears on your video.
- **Modern Aesthetics**: Sleek, unobtrusive design that complements your footage.
- **Easy to Use**: User-friendly CLI tool for quick and effortless video enhancement.
- **Performance Insights**: Visualize your ride data for better performance analysis.

Get ready to elevate your cycling videos with VidCycle! 🚴💨

## Installation

Welcome to the easy step-by-step installation process for VidCycle, Let's get you set up and ready to transform your rides into captivating stories.

#### Step 1: Get the Essentials
- **Install [ffmpeg](https://ffmpeg.org/)**: This is a powerful tool that VidCycle uses for video processing.
- **Install [Python3](https://www.python.org/downloads/)**: Make sure you have Python3 on your system, as it's the heart of VidCycle.

#### Step 2: Get the VidCycle Code
- **Clone the VidCycle Repository**: Grab the latest version of VidCycle from our repository to ensure you have all the cool features.

#### Step 3: Install Python Packages
- **Run the Installation Command**: In your command line, type `pip install -r requirements.txt` to install all the necessary Python packages VidCycle needs to run smoothly.

#### Step 4: Ready, Set, Go!
- **You're All Set!**: Congratulations, you've successfully installed VidCycle! You're now ready to start adding awesome data overlays to your cycling videos.

## VidCycle Usage Guide

Here's a clear and simple guide to help you create those amazing videos with data overlays. Let's dive in!

#### Step 1: Start Your Journey
- **Record Your Ride**: Use any camera you like. Just make sure the time on your camera is correctly set to match real-world time.

#### Step 2: Mark Your Start Moments
- **Use the Lap Button**: While recording, press the lap button on your Garmin bike computer whenever you want to highlight a specific moment.
- **Catch the Beep**: Ensure your camera's microphone can pick up the beep from your Garmin. This beep is crucial as it serves as the 'action' sync point to align your Garmin data with your video.

#### Step 3: Save Your Adventure
- **Transfer Files to Your Computer**: After your ride, load both the video files and the Garmin FIT file onto your computer.

#### Step 4: Run VidCycle
- **Align and Process**: Execute `python3 main.py` with the necessary parameters to align your video with the lap times from your Garmin. Need help with parameters? Just run `python3 main.py --help` for guidance.

#### Step 5: Enjoy the Result
- **Relive Your Ride**: After the program finishes processing, sit back and enjoy your cycling journey with all the key data beautifully integrated into your video.

## Example Video

Here's an example video that gives you an idea of what you can create.

[![Fat Cake - Hawk Hill - August 22, 2023](https://img.youtube.com/vi/KuYK_RrEdTI/0.jpg)](https://www.youtube.com/watch?v=KuYK_RrEdTI)

## Use case:

Video file path: $HOME/Downloads/18416701210/YDXJ0273.MP4 
Video start time:   2025-03-30 10:02:30+00:00
Garmin start time:  2025-03-30 09:05:14+00:00

The start time of the video is 1 hour ahead and the minutes are not accurate...

The lap time beep in the video occurs after 7 seconds.
The lap time in the fit file is between 0 and 240 seconds (reference time of the video).

The following command solves the time problems and generates the video.

```
../venv/bin/python main.py --fit-file $HOME/Downloads/18416701210/18416701210_ACTIVITY.fit --video-files $HOME/Downloads/18416701210/YDXJ0273.MP4 --video-output-path $HOME/Downloads/18416701210/out.mp4 --video-lap-time-in-secs 7 --render-config-file ./configs/1k-map-and-stats.json --lap-time-search-window-in-secs 0 240 --video-offset-start-in-secs 0 --video-hours-delta -1
```
