# Copyright 2016 James Sergeant. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import rosbag
import tensorflow as tf

import threading

import numpy as np
import os
import fnmatch

import cv2
from cv_bridge import CvBridge

def find_files(directory, pattern='*.bag'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_rosbag(directory, topics, bridge):
    '''Generator that yields images from ROSbag files'''
    files = find_files(directory)
    for filename in files:
        bag = rosbag.Bag(filename)
        for topic, msg, t in bag.read_messages(topics=topics):
            image = bridge.imgmsg_to_cv2(msg)
            yield image
        bag.close()

class ROSBagImageDataset(object):

    def __init__(self,
                coord,
                directory=None,
                topics=None,
                resize_dims=None,
                capacity=200, min_after_dequeue=100):

        self._topics = topics
        self._directory = directory
        self._placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self._cv_bridge = CvBridge()
        self._resize_dims = resize_dims
        self._coord = coord
        self.threads = []

        if self._resize_dims is not None:
            shapes = [(self._resize_dims[1],self._resize_dims[0],3)]
        else:
            shapes = [(None, None, 3)]

        self.queue = tf.FIFOQueue(capacity, min_after_dequeue,
                                         ['float32'],
                                         shapes=shapes)
        self.enqueue = self.queue.enqueue([self._placeholder])

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        while not stop:
            iterator = load_rosbag(self._directory, self._topics, self._cv_bridge)
            for image in iterator:
                if self._coord.should_stop():
                    stop = True
                    break
                if self._resize_dims is not None:
                    image = cv2.resize(image,self._resize_dims,
                                interpolation = cv2.INTER_CUBIC)
                sess.run(self.enqueue,
                         feed_dict={self._placeholder: image})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
