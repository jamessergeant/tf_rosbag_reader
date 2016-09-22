

from __future__ import print_function

import argparse
import time

import tensorflow as tf

from tf_rosbag_reader import ROSBagImageDataset

BATCH_SIZE = 1
NUM_STEPS = 10
DATA_DIRECTORY = '/home/james/co/tf_rosbag_reader/test'
TOPICS = ['/realsense/rgb/image_raw']

def get_arguments():
    parser = argparse.ArgumentParser(description='Test script')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='Directory containing bag files')
    parser.add_argument('--topics', type=tuple, default=TOPICS,
                        help='Image topics to extract from rosbag')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of steps')
    parser.add_argument('--gpu_fraction', type=float, default=0.1,
                        help='Specify amount of GPU memory to allocate')
    return parser.parse_args()

def main():
    args = get_arguments()

    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        reader = ROSBagImageDataset(
            coord,
            args.data_dir,
            topics=args.topics,
            resize_dims=(640,480))
        image_batch = reader.dequeue(args.batch_size)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options))

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    try:
        for step in range(0, args.num_steps):
            start_time = time.time()
            batch = sess.run(image_batch)
            print(batch.shape)
            duration = time.time() - start_time
            print('step %d, (%.3f sec/step)' % (step, duration))

    finally:
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
