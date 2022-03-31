import sys
import os
from os import path
from converter import convert_to_image


class ArgumentError(Exception):
    __module__ = Exception.__module__

    def __init__(self, message=''):
        super().__init__(message)


class DirectoryNotFoundError(Exception):
    __module__ = Exception.__module__

    def __init__(self, message=''):
        super().__init__(message)


def save_file(file_path, out_dir):
    if path.isfile(file_path):
        if '/' in file_path:
            filename = file_path.split('/')[-1].split('.')[0]
        elif '\\' in file_path:
            filename = file_path.split('\\')[-1].split('.')[0]
        else:
            filename = file_path.split('.')[0]

        dst = path.join(out_dir, '.'.join([filename, 'png']))

        with open(file_path, 'rb') as file:
            content = file.read()

        convert_to_image(256, content, dst)

    else:
        raise FileNotFoundError(f'"{file_path}" does not exists')


def save_directory(directory, out_dir):
    if path.isdir(directory):
        files_paths = os.listdir(directory)

        if '.DS_Store' in files_paths:
            files_paths.remove('.DS_Store')
        if '.git' in files_paths:
            files_paths.remove('.git')

        for file_path in files_paths:
            file_path = path.join(directory, file_path)
            save_file(file_path, out_dir)
    else:
        raise DirectoryNotFoundError(f'"{directory}" does not exists')


def save_directory_recursive(directory, out_dir):
    dir_content = os.listdir(directory)

    if '.DS_Store' in dir_content:
        dir_content.remove('.DS_Store')
    if '.git' in dir_content:
        dir_content.remove('.git')

    files_paths = [data_path for data_path in dir_content if '.' in data_path]
    dirs_paths = [data_path for data_path in dir_content if data_path not in files_paths]

    for file_path in files_paths:
        file_path = path.join(directory, file_path)
        save_file(file_path, out_dir)
    for dir_path in dirs_paths:
        parent_out_dir = out_dir
        out_dir = path.join(out_dir, dir_path)
        os.mkdir(out_dir)
        dir_path = path.join(directory, dir_path)
        save_directory_recursive(dir_path, out_dir)
        out_dir = parent_out_dir


def save():
    args = sys.argv

    if len(args) < 2:
        raise ArgumentError('No file or directory specified')
    if len(args) < 3:
        raise ArgumentError('No Output directory specified')

    args = args[1:]

    out_dir = args[1]

    if not path.isdir(out_dir):
        os.mkdir(out_dir)

    if len(args) > 2:
        if '-r' in args:
            args.remove('-r')

            parent_directory = args[0]
            save_directory_recursive(parent_directory, out_dir)
        else:
            raise ArgumentError(f'{args[2]} is not an acceptable argument')

    elif ('/' in args[0][-1]) or ('\\' in args[0][-1]):
        directory = args[0]

        save_directory(directory, out_dir)

    else:
        file_path = args[0]

        save_file(file_path, out_dir)


if __name__ == '__main__':
    save()
