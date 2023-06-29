import os

gpu_id = 1

def run_synthetic_data_eval_fig6():
    """
    Reproduce the result on synthetic data
    The result is consistent with Fig. 6 in the main paper
    """
    folder_paths = ['./data/synthetic_data/Buddha',
                    './data/synthetic_data/Bunny',
                    './data/synthetic_data/Tent']

    commit_id = 'TPAMI_submit_synthetic_eval'
    for data_dir in folder_paths:
        os.system("python ./train_nearPS_linux.py --data_folder {} --gpu_id {} --code_id {} --custom_depth_offset 3.0 --img_name img_sv_albedo.npy".format(
            data_dir, gpu_id, commit_id))



def run_finite_diff_eval_fig7():
    """
    Reproduce the result w/wo the analytical difference
    The result is consistent with Fig. 7 in the main paper
    """
    folder_paths = ['./data/synthetic_data/Buddha',
                    './data/synthetic_data/Bunny',
                    './data/synthetic_data/Tent']

    commit_id = 'TPAMI_submit_{}_difference'
    for diff_type in ['analytical', 'finite']:
        for data_dir in folder_paths:
            os.system("python ./train_nearPS_linux.py --data_folder {} --gpu_id {} --code_id {} --difference {} --custom_depth_offset 3.0 --img_name img_sv_albedo.npy".format(
                data_dir, gpu_id, commit_id.format(diff_type), diff_type))


def run_initial_depth_eval_fig8():
    """
    Reproduce the result on the sensitivity of depth initialization
    The result is consistent with Fig. 8 in the main paper
    """
    data_dir = './data/synthetic_data/Bear'

    commit_id = 'TPAMI_submit_depth_init_{}m'
    for init_depth in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
        os.system("python ./train_nearPS_linux.py --data_folder {} --gpu_id {} --code_id {} --custom_depth_offset {} --img_name img_sv_albedo.npy".format(
            data_dir, gpu_id, commit_id.format(init_depth), init_depth))


def run_real_data_eval_fig11():
    """
    Reproduce the result on real captured data
    The result is consistent with Fig. 11 in the main paper
    """
    folder_paths = ['./data/real_data/Angel',
                    './data/real_data/Plato',
                    './data/real_data/Stair']

    commit_id = 'TPAMI_submit_real_eval'
    for data_dir in folder_paths:
        os.system("python ./train_nearPS_linux.py --data_folder {} --gpu_id {} --code_id {} --custom_depth_offset 0.4 --img_name imgs_flir.npy".format(
            data_dir, gpu_id, commit_id))


if __name__ == '__main__':

    gpu_id = 1
    run_synthetic_data_eval_fig6()
    run_finite_diff_eval_fig7()
    run_initial_depth_eval_fig8()
    run_real_data_eval_fig11()