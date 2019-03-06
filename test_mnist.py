# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import argparse
import time

import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler
import six

# random seed must set before configuring the network.
fluid.default_startup_program().random_seed = 1

def parse_args():
    parser = argparse.ArgumentParser("mnist model benchmark.")
    parser.add_argument(
        '--batch_size', type=int, default=128, help='The minibatch size.')
    parser.add_argument(
        '--iterations', type=int, default=35, help='The number of minibatches.')
    parser.add_argument(
        '--pass_num', type=int, default=1, help='The number of passes.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--parallel_mode',
        action='store_true',
        help='Use parallel_mode or not.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def mnist_net(data):
    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=data,
        filter_size=5,
        num_filters=20,
        pool_size=2,
        pool_stride=2,
        act="relu")
    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=5,
        num_filters=50,
        pool_size=2,
        pool_stride=2,
        act="relu")
    predict = fluid.layers.fc(
        input=conv_pool_2,
        size=10,
        act="softmax")

    return predict


def eval_test(exe, batch_acc, batch_size_tensor, inference_program):
    test_reader = paddle.batch(
        paddle.dataset.mnist.test(), batch_size=args.batch_size)
    test_pass_acc = fluid.average.WeightedAverage()
    for batch_id, data in enumerate(test_reader()):
        img_data = np.array(
            [x[0].reshape([1, 28, 28]) for x in data]).astype('float32')
        y_data = np.array([x[1] for x in data]).reshape(
            [len(img_data), 1]).astype("int64")

        acc, weight = exe.run(inference_program,
                              feed={'pixel': img_data,
                                    'label': y_data},
                              fetch_list=[batch_acc, batch_size_tensor])
        test_pass_acc.add(value=acc, weight=weight)
        pass_acc = test_pass_acc.eval()
    return pass_acc


def run_mnist(args):
    # Input data
    images = fluid.layers.data(name='pixel', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # Train program
    predict = mnist_net(images)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

    # Evaluator
    batch_size_tensor = fluid.layers.create_tensor(dtype='int64')
    batch_acc = fluid.layers.accuracy(
        input=predict, label=label, total=batch_size_tensor)

    # inference program
    inference_program = fluid.default_main_program().clone(for_test=True)

    # Optimization
    opt = fluid.optimizer.AdamOptimizer(
        learning_rate=0.001, beta1=0.9, beta2=0.999)
    opt.minimize(avg_cost)

    # Initialize executor
    place = fluid.CPUPlace() if args.device == 'CPU' else fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    # Parameter initialization
    exe.run(fluid.default_startup_program())

    # compile the Program
    compiled_program = compiler.CompiledProgram(fluid.default_main_program())
    if args.parallel_mode:
        compiled_program.with_data_parallel(loss_name=avg_cost.name)

    # Reader
    train_reader = paddle.batch(
        paddle.dataset.mnist.train(), batch_size=args.batch_size)

    accuracy = fluid.average.WeightedAverage()
    for pass_id in range(args.pass_num):
        accuracy.reset()
        pass_start = time.time()
        every_pass_loss = []
        for batch_id, data in enumerate(train_reader()):
            img_data = np.array(
                [x[0].reshape([1, 28, 28]) for x in data]).astype('float32')
            y_data = np.array([x[1] for x in data]).reshape(
                [len(img_data), 1]).astype("int64")

            loss, acc, weight = exe.run(
                compiled_program,
                feed={'pixel': img_data,
                      'label': y_data},
                fetch_list=[avg_cost, batch_acc, batch_size_tensor])

            # The accuracy is the accumulation of batches, but not the current batch.
            accuracy.add(value=acc, weight=weight)
            every_pass_loss.append(loss)
            print("Pass = %d, Iter = %d, Loss = %s, Accuracy = %s" %
                  (pass_id, batch_id, loss, acc))
        pass_end = time.time()

        train_avg_acc = accuracy.eval()
        train_avg_loss = np.mean(every_pass_loss)
        test_avg_acc = eval_test(exe, batch_acc, batch_size_tensor,
                                 inference_program)

        print(
            "pass=%d, train_avg_acc=%f,train_avg_loss=%f, test_avg_acc=%f, elapse=%f"
            % (pass_id, train_avg_acc, train_avg_loss, test_avg_acc,
               (pass_end - pass_start)))
        # Note: The following logs are special for CE monitoring.
        # Other situations do not need to care about these logs.
        print("kpis	train_acc	%f" % train_avg_acc)
        print("kpis	train_cost	%f" % train_avg_loss)
        print("kpis	test_acc	%f" % test_avg_acc)
        print("kpis	train_duration	%f" % (pass_end - pass_start))


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    run_mnist(args)
