import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
import cv2
from Tools.deblurs_tools import deblur_image, viz_image, feature_orb
from cv_bridge import CvBridge

import numpy as np


class ImageSubscriber(Node):
    def __init__(self, num):
        super().__init__('image_subscriber')
        self.num = num
        self.cv_bridge = CvBridge()

        # Subscription to raw image topic
        self.subscription = self.create_subscription(
            CompressedImage,
            f'/boson{self.num}/image_raw/compressed',
            self.image_callback,
            10
        )
        self.subscription  # Prevent unused variable warning

        # A placeholder to store the latest image frame
        self.latest_image = None

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        color_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')

        # Store the latest image for processing
        self.latest_image = color_image
        self.timestamp = msg.header.stamp
        self.frame_id = msg.header.frame_id
        self.get_logger().info('Received a new image')


class ImageProcessor(Node):
    def __init__(self, num):
        super().__init__('image_processor')
        print("start", num)
        self.num = num
        # Publisher for the compressed image topic
        # if self.num == '0':
        #     print("num0")
        self.publisher = self.create_publisher(
            CompressedImage,
            f'/boson{self.num}/filter_image/compressed',
            10
        )
        # elif self.num == '1':
        #     print("num1")
        #     self.publisher1 = self.create_publisher(
        #         CompressedImage,
        #         f'/boson{self.num}/filter_image/compressed',
        #         10
        #     )

    def process_and_publish(self, image_subscriber):
        if image_subscriber.latest_image is not None:
            # modify image
            modified_image = deblur_image(image_subscriber.latest_image, 10, 10, kernel_type='two_dim', K_value=0.005)
            # Compress the image to JPEG

            # write to topic
            # Compress the image to JPEG format
            success, compressed_image = cv2.imencode('.jpg', modified_image)

            if not success:
                self.get_logger().error('Failed to compress image!')
                return

            # Create a CompressedImage message
            new_msg = CompressedImage()
            new_msg.format = 'jpeg'  # Specify the format (e.g., 'jpeg' or 'png')
            new_msg.data = compressed_image.tobytes()  # Convert the compressed image to raw bytes
            new_msg.header.stamp = image_subscriber.timestamp
            new_msg.header.frame_id = image_subscriber.frame_id
            # Publish the compressed image
            self.publisher.publish(new_msg)
            self.get_logger().info(f'Published a compressed image--{self.num}')


def main(args=None):
    rclpy.init(args=args)

    # Instantiate the nodes
    image_subscriber0 = ImageSubscriber(0)
    image_processor0 = ImageProcessor(0)
    image_subscriber1 = ImageSubscriber(1)
    image_processor1 = ImageProcessor(1)

    # Multi-threaded executor for parallel processing
    executor = MultiThreadedExecutor()

    # Add nodes to the executor
    executor.add_node(image_subscriber0)
    executor.add_node(image_processor0)
    executor.add_node(image_subscriber1)
    executor.add_node(image_processor1)

    try:
        while rclpy.ok():
            # Let executor handle callbacks
            executor.spin_once(timeout_sec=0.1)

            # Trigger image processing and publishing in parallel
            image_processor0.process_and_publish(image_subscriber0)
            image_processor1.process_and_publish(image_subscriber1)


    except KeyboardInterrupt:
        image_subscriber0.get_logger().info('0-Shutting down...')
        image_subscriber1.get_logger().info('1-Shutting down...')

    finally:
        # Clean up
        executor.shutdown()
        image_subscriber0.destroy_node()
        image_processor0.destroy_node()
        image_subscriber1.destroy_node()
        image_processor1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
