import os
import shutil
import random


#Grouping user label
#@path: user lable root directory
#@size: group size
def user_grouping(path, size):
    base_label_path = path
    group_label_path = './input/'

    label_dir_list = [d for d in os.listdir(base_label_path) if os.path.isdir(path + d)]

    if size == 0:
        size = 1

    if size > len(label_dir_list):
        size = 1

    num_group = len(label_dir_list)/size
    if len(label_dir_list) % size != 0:
        num_group += 1

    group_list = [label_dir_list[i*size:i*size+size] for i in range(num_group)]

    #print(group_list)

    if os.path.exists(group_label_path + 'groups'):
        shutil.rmtree(group_label_path + 'groups')

    if os.path.exists(group_label_path + 'groups') == False:
        os.mkdir(group_label_path + 'groups')

    # Create group directories
    for i in range(len(group_list)):
        group_name = 'group' + str(i)

        if os.path.exists(group_label_path + 'groups/' + group_name) == False:
            os.mkdir(group_label_path + 'groups/' + group_name)

        for user_name in group_list[i]:
            user_dir = group_label_path + 'groups/' + group_name + '/' + user_name
            shutil.copytree(base_label_path + user_name, user_dir + '/')

        if os.path.exists(group_label_path + 'groups/' + group_name + '/Unknown') == False:
            os.mkdir(group_label_path + 'groups/' + group_name + '/Unknown')

    #Sample Unknown class for each group
    group_id = 0

    for group in group_list:
        names = [n for n in group]
        unknowns = [u for u in label_dir_list if u not in names]

        print(names)
        print(unknowns)

        print('##########################')
        for unknown in unknowns:
            list_in_unknown = os.listdir(base_label_path + unknown)
            #print(path + unknown)
            #print(list_in_unknown)
            random.shuffle(list_in_unknown)
            #print(list_in_unknown)
            list_in_unknown = random.sample(list_in_unknown, 10)
            #print(list_in_unknown)

            for file_name in list_in_unknown:
                shutil.copy(base_label_path + unknown + '/' + file_name, group_label_path + 'groups/group' + str(group_id) + '/Unknown/')

        group_id = group_id + 1


def random_sample(path, ratio, out_dir, num_batch=4):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    os.mkdir(out_dir)

    print(path)

    label_dir_list = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    total_samples = 0
    num_group = len(label_dir_list)

    for label in label_dir_list:
        label_dir = os.path.join(path, label)

        total_samples += len(os.listdir(label_dir))

    print('Total Samples #: ' + str(total_samples))

    total_samples = total_samples // num_batch
    total_samples = total_samples * num_batch
    target_samples = total_samples * ratio // 100

    print('Ratio: ' + str(ratio) + '%')
    print('Target Samples #: ' + str(target_samples))
    target_samples_per_group = target_samples // num_group
    target_samples_per_group = num_batch * ((target_samples_per_group + num_batch - 1) // num_batch)
    print('Target Samples per Group #: ' + str(target_samples_per_group))

    for label in label_dir_list:
        label_dir = os.path.join(path, label)
        samples_in_label = os.listdir(label_dir)

        random.shuffle(samples_in_label)
        samples_in_label = random.sample(samples_in_label, target_samples_per_group)

        target_dir = os.path.join(out_dir, label)
        os.mkdir(target_dir)

        for target_file in samples_in_label:
            src = os.path.join(label_dir, target_file)
            dst = os.path.join(target_dir, target_file)
            shutil.copy(src, dst)


random_sample(os.path.abspath('input'), 10, os.path.abspath('class_input'))
