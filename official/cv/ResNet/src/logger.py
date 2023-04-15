# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =======================================================================================
"""Custom Logger."""
import os
import sys
import logging


class LOGGER(logging.Logger):
    """
    Logger.

    Args:
         logger_name: String. Logger name.
         rank: Integer. Rank id.
    """
    def __init__(self, logger_name, rank=0, param_server=False):
        super(LOGGER, self).__init__(logger_name)
        self.rank = rank
        if rank % 8 == 0 or param_server or self.use_server():
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
            console.setFormatter(formatter)
            self.addHandler(console)

    @staticmethod
    def use_server():
        worked = os.getenv('MS_WORKER_NUM', None)
        server = os.getenv('MS_SERVER_NUM', None)
        if worked is not None and server is not None:
            return True
        return False

    def setup_logging_file(self, log_dir):
        """Setup logging file."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_name = 'log.txt'
        self.log_fn = os.path.join(log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        fh.setFormatter(formatter)
        self.addHandler(fh)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info('Args:')
        args_dict = vars(args)
        for key in args_dict.keys():
            self.info('--> %s: %s', key, args_dict[key])
        self.info('')

    def important_info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and self.rank == 0:
            line_width = 2
            important_msg = '\n'
            important_msg += ('*'*70 + '\n')*line_width
            important_msg += ('*'*line_width + '\n')*2
            important_msg += '*'*line_width + ' '*8 + msg + '\n'
            important_msg += ('*'*line_width + '\n')*2
            important_msg += ('*'*70 + '\n')*line_width
            self.info(important_msg, *args, **kwargs)


def get_logger(path, rank, param_server=False):
    """Get Logger."""
    logger = LOGGER('resnet', rank, param_server=param_server)
    logger.setup_logging_file(os.path.join(path, 'rank_' + str(rank)))
    return logger
