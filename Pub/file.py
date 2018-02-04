import os
import random

# get file name and type in the directory
# walk returns 3 items: current director, directories in current directory and files in current direc
# and recursion unfold
def get_file_name(path):
    if os.path.exists(path):
        for root, dirs, files in os.walk(path):
            # print(root) #current path
            # print(dirs) #sub directories in current path
            # print(files) #all files in this path, directories not included
            return files
    return None


def file_exist(path, file_name):
    if not os.path.exists(path):
        return False
    for _, __, files in os.walk(path):
        if files.__contains__(file_name):
                return True
    return False


# return all files end up with suffix in file_path
def file_filter(file_path, suffix):
    lst = get_file_name(file_path)
    if lst:
        return [item for item in lst if item.find(suffix) != -1]
    print('Path Not Exists!')

def shuffle_file(path, num=1):
    names = get_file_name(path)
    total_num = len(names)
    shuffle_lst = []
    for i in range(num):
        shuffle_lst.append(random.choice(names))
    return shuffle_lst

if __name__ == '__main__':
    # get_file_name('E:\机器人\工作日志')
    # print(file_exist('E:/tmp/logs','events.out.tfevents.1492489548.PC201703231452'))
    # print(file_filter('E:/tmp/logs', '.PC201703231452'))
    print(shuffle_file('E:/PycharmProjects/V_CODE/Feasibility/data/ori/', 2))