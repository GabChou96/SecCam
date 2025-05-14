import time
import rclpy
from cv2 import magnitude
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge
from Tools.deblurs_tools import deblur_image, viz_image, feature_orb
import numpy as np
# from rclpy.qos import QoSProfile, ReliabilityPolicy
# import matplotlib.pyplot as plt
# from rclpy.executors import MultiThreadedExecutor
# import multiprocessing

# class ImageAndDepthSubscriber(Node):
#     def __init__(self):
#         super().__init__('image_and_depth_subscriber')
#         self.cv_bridge = CvBridge()
#         # Subscriber for compressed image topic (e.g., RGB camera)
#         self.camera_topic0 = '/boson0/image_raw/compressed'  # Replace with your RGB image topic
#         self.image_subscription0 = self.create_subscription(
#             CompressedImage,
#             self.camera_topic0,
#             lambda msg: self.image_callback(msg, '0'),
#             10  # QoS
#         )
#         self.camera_topic1 = '/rectify_left/image_rect/compressed'  # Replace with your RGB image topic
#         self.image_subscription1 = self.create_subscription(
#             CompressedImage,
#             self.camera_topic1,
#             lambda msg: self.image_callback(msg, '1'),
#             10  # QoS
#         )
#
#         # Publisher for the filtered data
#         self.publisher0 = self.create_publisher(
#             CompressedImage,  # Message type of the new topic
#             '/boson0/filter_image/compressed',  # Replace with your desired filtered topic name
#             10  # QoS depth
#         )
#
#         # Publisher for the filtered data
#         self.publisher1 = self.create_publisher(
#             CompressedImage,  # Message type of the new topic
#             '/rectify_left/image_rect/compressed',  # Replace with your desired filtered topic name
#             10  # QoS depth
#         )
#
#     def image_callback(self, msg, cam_num):
#         self.get_logger().info('Received an Thermal image message')
#
#         # Decode the compressed image
#         try:
#             color_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
#             print(msg.header.stamp)
#             modified_image = deblur_image(color_image, 20, 10, kernel_type='two_dim', K_value=0.005)
#             cv2.imshow(f'Received Image_{cam_num}', np.hstack([color_image, modified_image]))
#             cv2.waitKey(1)
#             # write to topic
#             # Compress the image to JPEG format
#             # success, compressed_image = cv2.imencode('.jpg', modified_image)
#             #
#             # if not success:
#             #     self.get_logger().error('Failed to compress image!')
#             #     return
#             #
#             # # Create a CompressedImage message
#             # new_msg = CompressedImage()
#             # new_msg.format = 'jpeg'  # Specify the format (e.g., 'jpeg' or 'png')
#             # new_msg.data = compressed_image.tobytes()  # Convert the compressed image to raw bytes
#             # new_msg.header.stamp = msg.header.stamp
#             # new_msg.header.frame_id = msg.header.frame_id
#             #
#             # # Publish the message
#             # if cam_num == '0':
#             #     self.publisher0.publish(new_msg)
#             # if cam_num == '1':
#             #     self.publisher1.publish(new_msg)
#             # self.get_logger().info(f'Published compressed image to topic: compressed_image')
#
#
#         except Exception as e:
#             self.get_logger().error(f"Failed to decode image: {e}")



class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_and_depth_subscriber')
        self.cv_bridge = CvBridge()
        # Subscriber for compressed image topic (e.g., RGB camera)
        # self.camera_topic0 = '/boson0/image_raw/compressed'  # Replace with your RGB image topic
        self.camera_topic0 = '/boson0/image_raw/compressed'  # Replace with your RGB image topic
        self.image_subscription0 = self.create_subscription(
            CompressedImage,
            self.camera_topic0,
            lambda msg: self.image_callback(msg, '0'),
            10  # QoS
        )

        # self.camera_topic1 = '/boson1/image_raw/compressed'  # Replace with your RGB image topic
        self.camera_topic1 = '/boson1/image_raw/compressed'  # Replace with your RGB image topic
        self.image_subscription1 = self.create_subscription(
            CompressedImage,
            self.camera_topic1,
            lambda msg: self.image_callback(msg, '1'),
            10  # QoS
        )
        #
        # # Publisher for the filtered data
        # self.publisher0 = self.create_publisher(
        #     CompressedImage,  # Message type of the new topic
        #     f'/boson0/filter_image/compressed',  # Replace with your desired filtered topic name
        #     10  # QoS depth
        # )
        # self.publisher1 = self.create_publisher(
        #     CompressedImage,  # Message type of the new topic
        #     f'/boson1/filter_image/compressed',  # Replace with your desired filtered topic name
        #     10  # QoS depth
        # )

    def image_callback(self, msg, cam_num):
        self.get_logger().info('Received an Thermal image message')

        # Decode the compressed image
        try:
            color_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            print(msg.header.stamp)
            # modified_image = deblur_image(color_image, 10, 5, kernel_type='both', K_value=0.005)
            modified_image = deblur_image(color_image, 20, 10, kernel_type='two_dim', K_value=0.005)
            cv2.imshow(f'Received Image_{cam_num}', np.hstack([color_image, modified_image]))
            cv2.waitKey(1)
            # write to topic
            # Compress the image to JPEG format
            success, compressed_image = cv2.imencode('.jpg', modified_image)

            if not success:
                self.get_logger().error('Failed to compress image!')
                return

            # Create a CompressedImage message
            # new_msg = CompressedImage()
            # new_msg.format = 'jpeg'  # Specify the format (e.g., 'jpeg' or 'png')
            # new_msg.data = compressed_image.tobytes()  # Convert the compressed image to raw bytes
            # new_msg.header.stamp = msg.header.stamp
            # new_msg.header.frame_id = msg.header.frame_id
            # if cam_num == '0':
            #     self.publisher0.publish(new_msg)
            # else:
            #     self.publisher1.publish(new_msg)

            self.get_logger().info(f'Published compressed image to topic: compressed_image')

        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")



# def main(args=None):
#     rclpy.init(args=args)
#     node = ImageAndDepthSubscriber()
#
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Node stopped cleanly')
#     except Exception as e:
#         node.get_logger().error(f'Error in node: {e}')
#     finally:
#         cv2.destroyAllWindows()
#         node.destroy_node()
#         rclpy.shutdown()

# def main():
#     rclpy.init()
#
#     # Create two instances of the ImageProcessor for two topics
#     processor0 = ImageSubscriber('/boson0/image_raw/compressed', '0')
#     processor1 = ImageSubscriber('/boson1/image_raw/compressed', '1')
#
#     # Use MultiThreadedExecutor to run both nodes in parallel
#     executor = MultiThreadedExecutor()
#     executor.add_node(processor0)
#     executor.add_node(processor1)
#
#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         processor0.destroy_node()
#         processor1.destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Error in node: {e}')
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()


