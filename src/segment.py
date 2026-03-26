#!/usr/bin/python3

import os
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image as PILImage
from torchvision.transforms import transforms

import rclpy

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class TensorRTInference:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)

        self.engine = self.load_engine(engine_path)
        
        self.context = self.engine.create_execution_context()
        #allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        return engine

    def __del__(self):
        # Free CUDA memory
        for inp in self.inputs:
            inp.device.free()
        for out in self.outputs:
            out.device.free()

    class HostDeviceMem:
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

    def allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for i, tensor_name in enumerate(engine):
            # print("-----: ", engine[0], engine[1])
            # tensor_name = engine.get_tensor_name()
            # print("-------: ", tensor_name)
            size = trt.volume(engine.get_tensor_shape(tensor_name))
            # print("-------: ", size)
            dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
            # print("-------: ", dtype)

            #allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            #append the device buffer address to device bindings
            bindings.append(int(device_mem))

            #append to the appropriate i/o list
            if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append(self.HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(self.HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        #transfer input data to device
        np.copyto(self.inputs[0].host, input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0].device, self.inputs[0].host, self.stream)

        #set tensor address
        for i in range(self.engine.num_io_tensors):
            self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

        #run inference
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        #transfer predictions back
        cuda.memcpy_dtoh_async(self.outputs[0].host, self.outputs[0].device, self.stream)

        #synchronize the stream
        self.stream.synchronize()

        return self.outputs[0].host

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentationNode')
        #/camera/camera/color/image_raw
        self.subscriber = self.create_subscription(Image, '/camera/image_raw', self.img_callback, 10)
        self.publisher = self.create_publisher(Image, '/segmentation', 10)
        self.bridge = CvBridge()

        model_path = os.getenv("MODELS")
        engine_path = f'{model_path}field_station.trt'
        self.trt_inference = TensorRTInference(engine_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((352, 672))
        ])


    def img_callback(self, data):

        cv_img = self.bridge.imgmsg_to_cv2(data, desired_encoding='rgb8')
        
        if cv_img is not None:
            # model_path = os.getenv("MODELS")
            # engine_path = f'{model_path}segmentationModel.trt'
            # trt_inference = TensorRTInference(engine_path)

            # transform = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Resize((352, 672))
            # ])

            pil_img = PILImage.fromarray(cv_img)

            resized_img = self.transform(pil_img)
            resized_array = resized_img.numpy()

            inputs = resized_array
            
            output_data = self.trt_inference.infer(inputs)
            output_data = torch.tensor(output_data)

            reshaped_pred = output_data.reshape((3, 352, 672))
            reshaped_pred = F.softmax(reshaped_pred, dim=0)
            
            reshaped_pred = reshaped_pred.permute(1, 2, 0).numpy()

            pred = np.argmax(reshaped_pred, axis=-1)

            result = np.zeros_like(reshaped_pred)
            
            # for i in range(3):
            #     result[max_channel_idx == i, i] = 255
            # Background (class 0) - dark gray
            result[pred == 0] = [0, 0, 255]

            # Class 1 - bright red
            result[pred == 1] = [255, 0, 0]

            # Class 2 - bright green
            result[pred == 2] = [0, 255, 0]

            result = np.uint8(result)
            
            self.publisher.publish(self.bridge.cv2_to_imgmsg(result, encoding='rgb8'))



def main():
    rclpy.init()
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()