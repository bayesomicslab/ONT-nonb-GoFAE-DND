import sys
import h5py
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool

def compute_current_tombo(fast5_path):
    """Input: fast5 path
    Outout: Extract the current signal
    Note: This is only worked after Tombo ran on data"""
    # fast5_path = 'ffff0f5f-5a83-47e0-93c1-8a5057574ebf.fast5' # reverse
    # fast5_path = '0009a349-52eb-4500-a061-bec2b27858f7.fast5'  # forward
    f = h5py.File(fast5_path, 'r')
    tombo_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
    current = [e[0] for e in tombo_events]
    return np.array(current)

def compute_seq_tombo(fast5_path):
    """Input: fast5 path
    Outout: Extract the bases
    Note: This is only worked after Tombo ran on data"""
    f = h5py.File(fast5_path, 'r')
    tombo_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
    bases = ''.join([e[4].decode() for e in tombo_events])
    return bases

def compute_translocation_time_tombo(fast5_path):
    """Input: fast5 path
    Outout: Computes the translocation  signal of a fast5 file
    Note: This is only worked after Tombo ran on data"""
    # fast5_path = 'ffff0f5f-5a83-47e0-93c1-8a5057574ebf.fast5' # reverse
    # fast5_path = '0009a349-52eb-4500-a061-bec2b27858f7.fast5'  # forward
    f = h5py.File(fast5_path, 'r')
    sampling_rate = f['UniqueGlobalKey/channel_id'].attrs.get('sampling_rate')
    tombo_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
    moves = [e[3] for e in tombo_events]
    trans_sec = np.array(moves)/sampling_rate
    return trans_sec

def compute_raw_signal(fast5_path, start_id, win_size):
    """Input: fast5 path
    Outout: Computes raw signal"""
    # fast5_path = 'ffff0f5f-5a83-47e0-93c1-8a5057574ebf.fast5' # reverse
    # fast5_path = '0009a349-52eb-4500-a061-bec2b27858f7.fast5'  # forward
    f = h5py.File(fast5_path, 'r')
    
    read_no = list(f['Raw/Reads'])[0]
    raw = list(f['Raw/Reads/' + read_no + '/Signal'])

    # if we want to compute Tombo normalized raw
    # scale = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('scale')
    # shift = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('shift')
    # norm_raw = (raw - shift)/scale

    corrected_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
    re_idx = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs.get('read_start_rel_to_raw')

    # read_raw_norm = norm_raw[re_idx+corrected_events[start_id][2]:re_idx+corrected_events[start_id+win_size][2] + corrected_events[start_id+win_size][3]]
    read_raw = raw[re_idx+corrected_events[start_id][2]:re_idx+corrected_events[start_id+win_size][2]+corrected_events[start_id+win_size][3]]

    return read_raw


def prepare_input(thread):
    bed_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr/'
    fast5_folder = '/labs/Aguiar/non_bdna/human/na12878/indices'
    save_path = '/labs/Aguiar/non_bdna/human/na12878/translocations/'
    input_list = [[name, bed_path, fast5_folder, save_path] for name in os.listdir(bed_path) if '.bed' in name]
    pool = Pool(thread)
    pool.map(make_signal_for_chr_strand, input_list)


def make_signal_for_chr_strand(inp):
    filename, bed_path, fast5_folder, save_path = inp
    name = filename.split('.bed')[0]
    this_save_path = os.path.join(save_path, name)
    if not os.path.exists(this_save_path):
        os.mkdir(this_save_path)
    existing_files = [nn.split('.npy')[0] for nn in os.listdir(this_save_path) if '.npy' in nn]
    filtered_df = pd.read_csv(os.path.join(bed_path, filename), names=['chr', 'start', 'end', 'read_id', 'score', 'strand'],
                              index_col=False, sep='\t')
    index_file = pd.read_csv(os.path.join(fast5_folder, name + '.csv'), index_col=0)
    unique_df = filtered_df.drop_duplicates().reset_index(drop=True)
    unique_df = unique_df[unique_df['score'] >= 20].reset_index(drop=True)
    read_ids = list(unique_df['read_id'])
    tombo_computed = list(index_file.index)
    ready_reads = list(set(read_ids).intersection(set(tombo_computed)))
    need_computing = [ff for ff in ready_reads if ff not in existing_files]
    for rd_id in need_computing:
        fast5_path = index_file.loc[rd_id, 'path']
        trans_loc_signal = compute_translocation_time_tombo(fast5_path)
        np.save(os.path.join(this_save_path, rd_id), trans_loc_signal)


if __name__ == '__main__':
    
    if '-t' in sys.argv:
        threads = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        threads = 1
        
    prepare_input(threads)