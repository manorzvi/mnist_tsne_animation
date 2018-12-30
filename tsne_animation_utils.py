import os
import shutil

def create_dir_result(directory):
    current_directory = os.getcwd()
    print(current_directory)
    final_directory   = os.path.join(current_directory, directory)
    print(final_directory)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
        print(directory + ' created')
    else: #directory exist
        shutil.rmtree(final_directory)
        print(directory + ' removed')
        os.makedirs(final_directory)
        print(directory + ' created again')