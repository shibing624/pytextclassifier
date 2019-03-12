# -*- coding: utf-8 -*-
# Author: XuMing <xuming624@qq.com>
# Brief:

import logging
import os


def get_file_list(path, postfix, file_list):
    """
    get postfix filename under path
    :param path:
    :param postfix:
    :param file_list:
    :return:
    """
    temp_list = os.listdir(path)
    for f in temp_list:
        f_d = os.path.join(path, f)
        if os.path.isdir(f_d):  # directory need recursion
            get_file_list(f_d, postfix, file_list)
        else:
            if f_d.endswith(postfix):
                file_list.append(f_d)
    return None


def clear_directory(path):
    """
    clear the dir of path
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    try:
        cmd = 'rm %s/*' % path
        print(cmd)
        os.popen(cmd)
    except Exception as e:
        print("error: %s" % e)
        return False
    return True


def get_logger(name, log_file=None):
    """
    logger
    :param name: 模块名称
    :param log_file: 日志文件，如无则输出到标准输出
    :return:
    """
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if not log_file:
        handle = logging.StreamHandler()
    else:
        handle = logging.FileHandler(log_file)
    handle.setFormatter(format)
    logger = logging.getLogger(name)
    logger.addHandler(handle)
    logger.setLevel(logging.DEBUG)
    return logger
