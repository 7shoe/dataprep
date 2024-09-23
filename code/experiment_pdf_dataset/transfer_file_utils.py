from concurrent.futures import ProcessPoolExecutor
import os
import shutil
from pathlib import Path

def copy_file(src_dst_tuple):
    src, dst = src_dst_tuple
    shutil.copyfile(src, dst)

def parallel_copy_files(sampled_pdf_path_dict, dst_root):
    tasks = []
    for k in sampled_pdf_path_dict:
        dst_path = dst_root / k
        for s in sampled_pdf_path_dict[k]:
            dst_file_path_loc = dst_path / s.name
            tasks.append((s, dst_file_path_loc))

    with ProcessPoolExecutor() as executor:
        executor.map(copy_file, tasks)