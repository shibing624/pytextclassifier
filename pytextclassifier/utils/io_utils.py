# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:
"""

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
