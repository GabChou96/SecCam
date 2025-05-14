import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import threading
from queue import Queue
from Tools.deblurs_tools import deblur_image, viz_image, feature_orb



class MultiThreadedImageProcessor(Node):
    def __init__(self):
        super().__init__('multi_threaded_image_processor')
        self.bridge = CvBridge()

        # Create queues to hold images between threads
        self.image_queue = Queue()
        self.updated_image_queue = Queue()

        # ROS 2 subscriptions and publications
        self.camera_topic = '/boson0/image_raw/compressed'
        self.image_subscription0 = self.create_subscription(
            CompressedImage,
            self.camera_topic,
            self.read_image,
            10  # QoS
        )
        self.publisher = self.create_publisher(CompressedImage, '/boson0/filter_image/compressed', 10)

        # Thread initialization
        self.read_thread = threading.Thread(target=self.read_thread_function)
        self.update_thread = threading.Thread(target=self.update_thread_function)
        self.write_thread = threading.Thread(target=self.write_thread_function)

        self.running = True

        # Start threads
        self.read_thread.start()
        self.update_thread.start()
        self.write_thread.start()

    def read_thread_function(self):
        """Thread to read new images from the image topic."""
        while self.running:
            rclpy.spin_once(self, timeout_sec=0.1)  # Process ROS 2 callbacks

    def read_image(self, msg):
        """Callback to handle incoming images."""
        self.get_logger().info('Reading new image...')
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.image_queue.put(frame)  # Add the image to the processing queue

    def update_thread_function(self):
        """Thread to update images from the queue."""
        while self.running:
            if not self.image_queue.empty():
                image = self.image_queue.get()
                self.get_logger().info('Updating image...')
                updated_image = self.update_image(image)
                self.updated_image_queue.put(updated_image)

    def update_image(self, image):
        """Process the image (e.g., apply Gaussian blur)."""
        modified_image = deblur_image(image, 20, 10, kernel_type='two_dim', K_value=0.005)
        return modified_image

    def write_thread_function(self):
        """Thread to publish updated images."""
        while self.running:
            if not self.updated_image_queue.empty():
                updated_image = self.updated_image_queue.get()
                self.get_logger().info('Writing updated image...')
                success, compressed_image = cv2.imencode('.jpg', updated_image)
                if not success:
                    self.get_logger().error('Failed to compress image!')
                    return
                msg = CompressedImage()
                msg.format = 'jpeg'  # Specify the format (e.g., 'jpeg' or 'png')
                msg.data = compressed_image.tobytes()  # Convert the compressed image to raw bytes
                # msg.header.stamp = msg.header.stamp
                # msg.header.frame_id = msg.header.frame_id
                self.publisher.publish(msg)

    def destroy_node(self):
        """Cleanup on node shutdown."""
        self.running = False
        self.read_thread.join()
        self.update_thread.join()
        self.write_thread.join()
        super().destroy_node()


def main():
    rclpy.init()

    # Create and run the multi-threaded image processor node
    processor_node = MultiThreadedImageProcessor()

    try:
        # Keep the node running
        rclpy.spin(processor_node)
    except KeyboardInterrupt:
        processor_node.get_logger().info('Shutting down...')
    finally:
        processor_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
