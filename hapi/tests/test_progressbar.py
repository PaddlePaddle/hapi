# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import random
import time

from hapi.progressbar import ProgressBar


class TestProgressBar(unittest.TestCase):

    def prog_bar(self, num, loop_num, width):
        progbar = ProgressBar(num)
        values = [
            ['loss', 50.341673],
            ['acc', 0.00256],
        ]
        for i in range(loop_num):
            values[0][1] -= random.random() * 0.1
            values[1][1] += random.random() * 0.1
            if i % 10 == 0 or i == loop_num - 1:
                progbar.update(i, values)
            time.sleep(0.01)

    def test1(self):
        self.prog_bar(100, 100, 30)

    def test2(self):
        self.prog_bar(50, 100, 30)

    def test2(self):
        self.prog_bar(None, 50, 30)

if __name__ == '__main__':
    unittest.main()
