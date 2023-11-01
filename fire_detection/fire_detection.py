#!/usr/bin/env python3
# coding=utf-8

# import module

from pathlib import Path
import argparse

import cv2
import depthai
import numpy as np
from imutils.video import FPS

# import depthai SDK

from depthai_sdk import OakCamera, DetectionPacket, Visualizer, TextPosition

# Arguments Config

# Create Argparse Object for configuration
parser = argparse.ArgumentParser()

# add_argument ('posistional argument', 'option that take a value', 'action -> on/off tag', 'help -> instruction printed when help mode')
# command tag: no debug --> For running the Code even with some error (which is not recommoned) 

parser.add_argument(
    "-nd", "--no-debug", action="store_true", help="prevent debug output"
)

# comand tag: camera --> For running the Code with the oak camera view

parser.add_argument(
    "-cam",
    "--camera",
    action="store_true",
    help="Use DepthAI 4K RGB camera for inference (conflicts with -vid)",
)

# command tag: video --> For running the Code with video by certain file path

parser.add_argument(
    "-vid",
    "--video",
    type=str,
    help="The path of the video file used for inference (conflicts with -cam)",
)

# Create the namespace for the params config

args = parser.parse_args()

# open a gate for the debugging

debug = not args.no_debug

# Result with wrong params input

if args.camera and args.video:
    raise ValueError(
        'Command line parameter error! "-Cam" cannot be used together with "-vid"!'
    )
elif args.camera is False and args.video is None:
    raise ValueError(
        'Missing inference source! Use "-cam" to run on DepthAI cameras, or use "-vid <path>" to run on video files'
    )


# ************************************************************

# Function for return value

# with the camera frame

def to_planar(arr: np.ndarray, shape: tuple):
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()

# non used function

def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())


# result of the detetction

def to_tensor_result(packet):
    return {
        name: np.array(packet.getLayerFp16(name))
        for name in [tensor.name for tensor in packet.getRaw().tensors]
    }

# start the detection

def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    return x_out.tryGet()

# ************************************************************
# Class DepthAI -> class for define the function will be used

class DepthAI:

    # initial config of the vlass

    def __init__(
        self,
        file=None, # File -> for video mode
        camera=False, # Camera -> for camera mode
    ):
        print("Loading pipeline...")

        self.file = file
        self.camera = camera

        # Get FPS from the souce
        self.fps_cam = FPS()
        self.fps_nn = FPS()

        # Create and Start the pipeline for the depthAI Package 

        self.create_pipeline()
        self.start_pipeline()

        # Set the text size and line type

        self.fontScale = 1 if self.camera else 2
        self.lineType = 0 if self.camera else 3

    # Define the create piepeline function

    def create_pipeline(self):
        print("Creating pipeline...")

        # Set the pipeline function from depthai module

        self.pipeline = depthai.Pipeline()

        # If Camera is linked to the this device

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")

            # Settings for creating the pipeline for the camera

            self.cam = self.pipeline.create(depthai.node.ColorCamera)
            self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
            self.cam.setResolution(
                depthai.ColorCameraProperties.SensorResolution.THE_4_K
            )
            self.cam.setInterleaved(False)
            self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

            self.cam_xout = self.pipeline.create(depthai.node.XLinkOut)
            self.cam_xout.setStreamName("preview")
            self.cam.preview.link(self.cam_xout.input)

        self.create_nns()

        print("Pipeline created.")

    def create_nns(self):
        pass


    # Defince Loading of the fire detection module

    def create_nn(self, model_path: str, model_name: str, first: bool = False):
        """
        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """

        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")

        # Standard config for loading a pre-trained Neural Network

        # Netual Network with depthAI doc url: https://docs.luxonis.com/projects/api/en/latest/components/nodes/neural_network/
        # 
        model_nn = self.pipeline.create(depthai.node.NeuralNetwork)
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first and self.camera:
            print("linked cam.preview to model_nn.input")
            self.cam.preview.link(model_nn.input)
        else:
            model_in = self.pipeline.create(depthai.node.XLinkIn)
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.create(depthai.node.XLinkOut)
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    # defince a function to run the pipeline and run the camera frame if camera mode is on

    def start_pipeline(self):
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")

        self.start_nns()

        if self.camera:

            # Output the data through the camera

            self.preview = self.device.getOutputQueue(
                name="preview", maxSize=4, blocking=False
            )

    def start_nns(self):
        pass


    # Define a function to print the detection result

    # Print on the Top Left corner of the video OR camera frame
    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )

    # Edit should start here
    # Define a function to print the result within a box with on the detected object

    # ****************************************


    # For open the image frame

    def parse(self):
        if debug:
            self.debug_frame = self.frame.copy()

        self.parse_fun()

        if debug:
            cv2.imshow(
                "Camera_view",
                self.debug_frame,
            )
            self.fps_cam.update()
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                self.fps_cam.stop()
                self.fps_nn.stop()
                print(
                    f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
                )
                raise StopIteration()

    def parse_fun(self):
        pass

    # For open the video frame

    def run_video(self):
        cap = cv2.VideoCapture(str(Path(self.file).resolve().absolute()))
        while cap.isOpened():
            read_correctly, self.frame = cap.read()
            if not read_correctly:
                break

            try:
                self.parse()
            except StopIteration:
                break

        cap.release()

    # For open the Camera View

    def run_camera(self):
        while True:
            in_rgb = self.preview.tryGet()
            if in_rgb is not None:
                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                self.frame = (
                    in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                )
                self.frame = np.ascontiguousarray(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break

    # set the camera size                

    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v

    # 

    def run(self):
        self.fps_cam.start()
        self.fps_nn.start()
        if self.file is not None:
            self.run_video()
        else:
            self.run_camera()
        del self.device

# ================================================================

# Class Main
class Main(DepthAI):

    # init the class with DepthAI

    def __init__(self, file=None, camera=False):
        self.cam_size = (1000, 1000)
        super().__init__(file, camera)

    # Loading the fire detection module from module path

    def create_nns(self):
        self.create_nn("models/fire-detection_openvino_2021.2_5shave.blob", "fire")

    def start_nns(self):

        # Input queue, to send message from the host to the device (you can receive the message on the device with XLinkIn)

        self.fire_in = self.device.getInputQueue("fire_in", 4, False)

        # Output queue, to receive message on the host from the device (you can send the message on the device with XLinkOut)

        self.fire_nn = self.device.getOutputQueue("fire_nn", 4, False)

    def run_fire(self):

        # define the data type for fire detection

        labels = ["fire", "normal", "smoke"]
        w, h = self.frame.shape[:2]

        # Send Data to the module and get return data

        nn_data = run_nn(
            self.fire_in, # data in
            self.fire_nn, # data out
            {"Placeholder": to_planar(self.frame, (224, 224))}, # camera 
        )
        if nn_data is None:
            return
        
        # update the camera / video frame with the FPS

        self.fps_nn.update()

        # Get the result from the detection module
        # Data Type: Array -> 

        results = to_tensor_result(nn_data).get("final_result")

        # result = 0 -> fire, result = 1 -> normal, result = 2 -> smoke

        i = int(np.argmax(results))

        # Print the detection result
        # Get the detection tag from data
        label = labels[i]

        # If there is no fire detected, print noting
        if label == "normal":
            return
        else:
        # If FIRE or SMOKE detectedm print it out
        
            if results[i] > 0.5: # if detected % > 0.5

            # Should Edit Here if want to have UI change
                self.put_text(
                    f"{label}:{results[i]:.2f}",
                    (10, 25),
                    color=(0, 0, 255),
                    font_scale=1,
                )


    # Self Call for looping the main function

    def parse_fun(self):

        self.run_fire()


if __name__ == "__main__":

    # Get the user input type of use
    # If user input "-vid" as params call a video run

    if args.video:
        Main(file=args.video).run()
    else:

    # If user input "-cam" as params, call a camera run        
        Main(camera=args.camera).run()