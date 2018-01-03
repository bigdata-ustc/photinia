import os

import gridfs
import subprocess
import pymongo


def print_progress(current_loop,
                   num_loops,
                   msg='Processing',
                   interval=1000):
    """Print progress information in a line.

    :param current_loop: Current loop number.
    :param num_loops: Total loop count.
    :param msg: Message shown on the line.
    :param interval: Interval loops. Default is 1000.
    """
    if current_loop % interval == 0 or current_loop == num_loops:
        print('%s [%d/%d]... %.2f%%' % (msg, current_loop, num_loops, current_loop / num_loops * 100), end='\n')


def read_file(file_):
    parts = list()
    with open(file_, 'rb') as f:
        while True:
            buffer = f.read(10240)
            if len(buffer) == 0:
                break
            parts.append(buffer)
    return b''.join(parts)


def main():
    frames_dir = '/data/frames'
    with open('/data/episodes.txt', 'rt') as f:
        lines = f.readlines()
    dirs = [os.path.join(frames_dir, line.strip() + '.frames') for line in lines]
    with pymongo.MongoClient('192.168.1.3') as client:
        client['admin'].authenticate('root', 'SELECT * FROM users;')
        db = client['crystal']
        coll = db['frames']
        for i, dir_ in enumerate(dirs):
            print_progress(i + 1, len(dirs), 'Processing %s' % dir_, interval=1)
            #
            image_files = os.listdir(dir_)
            episode_name = os.path.basename(dir_).replace('.frames', '')
            count_exist = coll.find({'episode_name': episode_name}).count()
            if count_exist == len(image_files):
                print('Skip.')
                continue
            doc_list = list()
            for image_file in image_files:
                name = image_file
                image_file = os.path.join(dir_, image_file)
                image_bytes = read_file(image_file)
                doc = {
                    'name': name,
                    'episode_name': episode_name,
                    'image': image_bytes
                }
                doc_list.append(doc)
            if count_exist != 0:
                coll.delete_many({'episode_name': episode_name})
            coll.insert_many(doc_list)
            print('Insert.')
    return 0


if __name__ == '__main__':
    exit(main())
