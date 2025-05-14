import time

import rclpy
from cv2 import magnitude
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
import cv2
from cv_bridge import CvBridge
from Tools.deblurs_tools import deblur_image, viz_image, feature_orb
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
import matplotlib.pyplot as plt



class ImageAndDepthSubscriber(Node):
    def __init__(self):
        super().__init__('image_and_depth_subscriber')
        self.cv_bridge = CvBridge()
        self.i = 0
        # Subscriber for compressed image topic (e.g., RGB camera)
        self.camera_topic0 = '/boson0/image_raw/compressed'  # Replace with your RGB image topic
        self.image_subscription0 = self.create_subscription(
            CompressedImage,
            self.camera_topic0,
            lambda msg: self.image_callback(msg, '0'),
            10  # QoS
        )

        self.camera_topic1 = '/rectify_left/image_rect/compressed'  # Replace with your RGB image topic
        self.image_subscription1 = self.create_subscription(
            CompressedImage,
            self.camera_topic1,
            lambda msg: self.image_callback(msg, '1'),
            10  # QoS
        )
        self.left_count = 0
        self.right_count = 0
        self.tot_count = 1
        self.re_left_count = 0
        self.re_right_count = 0
        self.re_tot_count = 1

        # Subscriber for depth image topic
        # self.qos_policy = QoSProfile(depth=10,  # Queue size
        #                              reliability=ReliabilityPolicy.BEST_EFFORT)  # Change to match the publisher's reliability
        # self.depth_topic = '/ov_msckf/loop_depth'  # Replace with your depth image topic
        # self.depth_subscription = self.create_subscription(
        #     Image,
        #     self.depth_topic,
        #     self.depth_callback,
        #     self.qos_policy  # QoS
        # )
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.video_writer = cv2.VideoWriter("therm.avi", fourcc, 20, (1280, 512), False)
        # self.subscription  # Prevent unused variable warning

        # # Publisher for the filtered data
        # self.publisher0 = self.create_publisher(
        #     CompressedImage,  # Message type of the new topic
        #     '/boson0/filter_image/compressed',  # Replace with your desired filtered topic name
        #     10  # QoS depth
        # )
        # self.publisher1 = self.create_publisher(
        #     CompressedImage,  # Message type of the new topic
        #     '/boson1/filter_image/compressed',  # Replace with your desired filtered topic name
        #     10  # QoS depth
        # )





    def image_callback(self, msg, cam_num):
        self.get_logger().info('Received an Thermal image message')

        # Decode the compressed image
        try:
            color_image = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
            print(msg.header.stamp)

            # start= time.time_ns()/1000000
            modified_image = deblur_image(color_image, 10, 3, kernel_type='both', K_value=0.005)
            # print(time.time_ns()/1000000-start)
            left, right, c_left, c_right = feature_orb(color_image, modified_image)
            # Display the image
            # f= np.fft.fft2(color_image)
            # fshift = np.fft.fftshift(f)
            # magnitude_spec = 20*np.log(np.abs(fshift))
            # print(magnitude_spec.shape, color_image.shape)
            # cv2.imshow(f'Received Image_{cam_num}', magnitude_spec)
            # plt.imshow(magnitude_spec)
            # plt.show()
            # cv2.imshow("img", color_image)
            # cv2.imshow(f'Received Image_{cam_num}', np.hstack([cv2.Canny(color_image,100,200), cv2.Canny(modified_image, 150,170)]))
            cv2.imshow(f'Received Image_{cam_num}', np.hstack([color_image, modified_image]))
            #
            # cv2.imshow(f'Received Image_{cam_num}', np.hstack([left, right]))
            key = cv2.waitKey(1)  # Use 1 ms delay to keep the window responsive
            # self.video_writer.write(np.hstack([viz_image(color_image), modified_image]))
            if cam_num == '0':
                self.left_count += c_left
                self.right_count += c_right
                self.tot_count += 1

            elif cam_num == '1':
                self.re_left_count += c_left
                self.re_right_count += c_right
                self.re_tot_count += 1
            # #write to topic
            # Compress the image to JPEG format
            # success, compressed_image = cv2.imencode('.jpg', modified_image)
            #
            # if not success:
            #     self.get_logger().error('Failed to compress image!')
            #     return
            #
            # # Create a CompressedImage message
            # msg = CompressedImage()
            # msg.format = 'jpeg'  # Specify the format (e.g., 'jpeg' or 'png')
            # msg.data = compressed_image.tobytes()  # Convert the compressed image to raw bytes

            # # Publish the message
            # if cam_num == '0':
            #     self.publisher0.publish(msg)
            # if cam_num == '1':
            #     self.publisher1.publish(msg)
            # self.get_logger().info(f'Published compressed image to topic: compressed_image')
            # if key == ord('g'):
            #     cv2.imwrite(f"iim{self.i}.png", color_image)
            #     self.i+=1
            #     self.video_writer.release()

        except Exception as e:
            self.get_logger().error(f"Failed to decode image: {e}")

    # def depth_callback(self, msg):
    #     self.get_logger().info('Received a depth image message')
    #
    #     try:
    #         # Convert depth image data to a NumPy array
    #         depth_image = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
    #
    #         # Normalize depth values for visualization
    #         # depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    #         # depth_normalized = depth_normalized.astype(np.uint8)
    #
    #         # Display the depth image
    #         cv2.imshow("Depth Image", depth_image)
    #         key2 = cv2.waitKey(1000)  # Keep the window responsive
    #         if key2 == ord('g'):
    #             np.save('depth_loop1.npy', depth_image)
    #     except Exception as e:
    #         self.get_logger().error(f"Error processing depth image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageAndDepthSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped cleanly')
    except Exception as e:
        node.get_logger().error(f'Error in node: {e}')
    finally:
        print("without: ", node.left_count/node.tot_count, "with: ", node.right_count/node.tot_count)
        print("rectify: without: ", node.re_left_count / node.re_tot_count, "with: ", node.re_right_count / node.re_tot_count)
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()


