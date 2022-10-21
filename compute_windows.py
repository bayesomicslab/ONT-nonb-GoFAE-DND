import os
import sys
import pandas as pd
import numpy as np
from multiprocessing import Pool
from compute_translocation_time import compute_translocation_time_tombo, compute_current_tombo, compute_seq_tombo, compute_raw_signal
from utils import *
# import time

def make_inputs_for_creat_windows_per_chr(chrom):
    # t1 = time.perf_counter()
    translocation_path = '/labs/Aguiar/non_bdna/human/na12878/translocations'
    this_trans_location_path_pos = os.path.join(translocation_path, chrom + '+')
    this_trans_location_path_neg = os.path.join(translocation_path, chrom + '-')
    # this_trans_location_path = {'+': this_trans_location_path_pos, '-': this_trans_location_path_neg}

    windows_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping_0_based/' # new
    this_windows_path = os.path.join(windows_path, 'non_overlapping_windows_' + chrom + '.csv')
    windows_per_chr = pd.read_csv(this_windows_path)
    windows_per_chr = windows_per_chr.sort_values(by=['feature', 'win_start']).reset_index(drop=True)
    
    reads_path = '/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr'
    filtered_reads_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr'
    
    this_read_path_pos = os.path.join(reads_path, chrom + '+.csv')
    reads_per_chr_pos = pd.read_csv(this_read_path_pos, index_col=0)

    this_filtered_reads_path_pos = os.path.join(filtered_reads_path, chrom + '+.bed')
    filtered_reads_per_chr_pos = pd.read_csv(this_filtered_reads_path_pos, names=['chr', 'start', 'end', 'read_id', 'score', 'strand'], sep='\t')

    filtered_reads_per_chr_pos = filtered_reads_per_chr_pos[filtered_reads_per_chr_pos['score'] >= 20].reset_index(drop=True)
    final_filtered_reads_per_chr_pos = pd.merge(reads_per_chr_pos, filtered_reads_per_chr_pos, how='inner', on='read_id')
    final_filtered_reads_per_chr_pos = final_filtered_reads_per_chr_pos.drop(columns=['start', 'end', 'score'])

    this_read_path_neg = os.path.join(reads_path, chrom + '-.csv')
    reads_per_chr_neg = pd.read_csv(this_read_path_neg, index_col=0)

    this_filtered_reads_path_neg = os.path.join(filtered_reads_path, chrom + '-.bed')
    filtered_reads_per_chr_neg = pd.read_csv(this_filtered_reads_path_neg, names=['chr', 'start', 'end', 'read_id', 'score', 'strand'], sep='\t')

    filtered_reads_per_chr_neg = filtered_reads_per_chr_neg[filtered_reads_per_chr_neg['score'] >= 20].reset_index(drop=True)
    final_filtered_reads_per_chr_neg = pd.merge(reads_per_chr_neg, filtered_reads_per_chr_neg, how='inner', on='read_id')
    final_filtered_reads_per_chr_neg = final_filtered_reads_per_chr_neg.drop(columns=['start', 'end', 'score'])
    
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows_current_ipd_seq'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        
    this_save_path = os.path.join(save_path, chrom)
    if not os.path.exists(this_save_path):
        os.mkdir(this_save_path)
    n_windows_per_file = 100
    inputs = []
    labels = list(windows_per_chr['feature'].unique())
    for ll in labels:
        existing_files = []
        this_label_input = []
        this_label_df = windows_per_chr[windows_per_chr['feature'] == ll].reset_index(drop=True)
        this_label_save_path = os.path.join(this_save_path, ll)
        for i in range(len(this_label_df)):
            _, label, motif_start, motif_end, strand, _, motif_seq, _, win_start, win_end, direction, _, _ = this_label_df.loc[i, :]
            this_label_input.append([win_start, win_end, motif_seq, chrom, strand, direction, label, final_filtered_reads_per_chr_pos,
                                     final_filtered_reads_per_chr_neg, this_trans_location_path_pos, this_trans_location_path_neg])
        if os.path.exists(this_label_save_path):
            existing_files = [os.path.join(this_label_save_path, f) for f in os.listdir(this_label_save_path) if '.pkl.gz' in f]
        chunks = list(chunk_it(this_label_input, n_windows_per_file))
    
        inputs += [[ll, this_label_save_path, chunk] for chunk in chunks if os.path.join(this_label_save_path, get_chunk_name(chunk)) not in existing_files]
    # t2 = time.perf_counter()
    # print('preprocessing the input for chromosome', t2-t1)
    return inputs


def make_inputs_for_creat_windows_per_chr_strand(chrom, strand):
    """This funcion makes input for computation of the windows in pickle files of contatining 100 windows each
    It prepares all the input the computation function should have and chunck all the windows to packs of 100 windows"""
    
    translocation_path = '/labs/Aguiar/non_bdna/human/na12878/translocations'
    this_trans_location_path = os.path.join(translocation_path, chrom + strand)
    
    # windows_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping_interval_graph' # old
    windows_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping_0_based/' # new
    this_windows_path = os.path.join(windows_path, 'non_overlapping_windows_' + chrom + strand + '.csv')
    
    reads_path = '/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr'
    this_read_path = os.path.join(reads_path, chrom + strand+'.csv')
    
    filtered_reads_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr'
    this_filtered_reads_path = os.path.join(filtered_reads_path, chrom + strand+'.bed')
    
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows'
    this_save_path = os.path.join(save_path, chrom + strand)
    if not os.path.exists(this_save_path):
        os.mkdir(this_save_path)
    
    windows_per_chr = pd.read_csv(this_windows_path)
    windows_per_chr = windows_per_chr.sort_values(by=['feature', 'win_start']).reset_index(drop=True)
    reads_per_chr = pd.read_csv(this_read_path, index_col=0)
    filtered_reads_per_chr = pd.read_csv(this_filtered_reads_path, names=['chr', 'start', 'end', 'read_id', 'score', 'strand'], sep='\t')
    filtered_reads_per_chr = filtered_reads_per_chr[filtered_reads_per_chr['score'] >= 20].reset_index(drop=True)
    final_filtered_reads_per_chr = pd.merge(reads_per_chr, filtered_reads_per_chr, how='inner', on='read_id')
    final_filtered_reads_per_chr = final_filtered_reads_per_chr.drop(columns=['start', 'end', 'score'])
    
    n_windows_per_file = 100
    inputs = []
    labels = list(windows_per_chr['feature'].unique())
    for ll in labels:
        this_label_input = []
        this_label_df = windows_per_chr[windows_per_chr['feature'] == ll].reset_index(drop=True)
        this_label_save_path = os.path.join(this_save_path, ll)
        for i in range(len(this_label_df)):
            _, label, motif_start, motif_end, _, _, motif_seq, _, win_start, win_end, direction, _, _ = this_label_df.loc[i, :]
            this_label_input.append([win_start, win_end, motif_seq, chrom, strand, direction, label, final_filtered_reads_per_chr, this_trans_location_path])
        chunks = list(chunk_it(this_label_input, n_windows_per_file))
        inputs += [[ll, this_label_save_path, chunk] for chunk in chunks]
    return inputs


def make_inputs_for_creat_windows_per_chr_strand_control(chrom, strand):
    """This funcion makes input for computation of the windows in pickle files of contatining 100 windows each
    It prepares all the input the computation function should have and chunck all the windows to packs of 100 windows
    This is for control windows."""
    translocation_path = '/labs/Aguiar/non_bdna/human/na12878/translocations'
    this_trans_location_path = os.path.join(translocation_path, chrom + strand)
    
    # windows_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping_interval_graph' # old
    windows_path = '/labs/Aguiar/non_bdna/annotations/windows/control_conservative/' # these are not 0 based so we have to change convert them
    this_windows_path = os.path.join(windows_path, 'control_conservative_' + chrom + strand + '.csv')
    
    reads_path = '/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr'
    this_read_path = os.path.join(reads_path, chrom + strand+'.csv')
    
    filtered_reads_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr'
    this_filtered_reads_path = os.path.join(filtered_reads_path, chrom + strand+'.bed')
    
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows'
    this_save_path = os.path.join(save_path, chrom + strand)
    if not os.path.exists(this_save_path):
        os.mkdir(this_save_path)
        
    windows_per_chr = pd.read_csv(this_windows_path, index_col=0)
    reads_per_chr = pd.read_csv(this_read_path, index_col=0)
    filtered_reads_per_chr = pd.read_csv(this_filtered_reads_path, names=['chr', 'start', 'end', 'read_id', 'score', 'strand'], sep='\t')
    filtered_reads_per_chr = filtered_reads_per_chr[filtered_reads_per_chr['score'] >= 20].reset_index(drop=True)
    final_filtered_reads_per_chr = pd.merge(reads_per_chr, filtered_reads_per_chr, how='inner', on='read_id')
    final_filtered_reads_per_chr = final_filtered_reads_per_chr.drop(columns=['start', 'end', 'score'])

    n_windows_per_file = 100
    inputs = []
    labels = list(windows_per_chr['feature'].unique())
    for ll in labels:
        this_label_input = []
        this_label_df = windows_per_chr[windows_per_chr['feature'] == ll].reset_index(drop=True)
        this_label_save_path = os.path.join(this_save_path, ll)
        
        for i in range(len(this_label_df)):
            _, label, motif_start, motif_end, _, _, win_start, win_end = this_label_df.loc[i, :]
            motif_seq = ''
            motif_start -= 1
            motif_end -= 1
            win_start -= 1
            win_end -= 1
            this_label_input.append([win_start, win_end, motif_seq, chrom, strand, label, final_filtered_reads_per_chr, this_trans_location_path])
        chunks = list(chunk_it(this_label_input, n_windows_per_file))
        inputs += [[ll, this_label_save_path, chunk] for chunk in chunks]
    return inputs


def make_inputs_for_creat_continous_windows_ch(this_inp):
    chrom, query_len, n_file = this_inp
    translocation_path = '/labs/Aguiar/non_bdna/human/na12878/translocations'
    this_trans_location_path_pos = os.path.join(translocation_path, chrom + '+')
    this_trans_location_path_neg = os.path.join(translocation_path, chrom + '-')
    
    windows_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/continuous_windows'
    next_path = os.path.join(windows_path, chrom + '_completed')
    if not os.path.exists(next_path):
        os.mkdir(next_path)

    this_windows_path = os.path.join(windows_path, chrom, 'continuous_' + chrom + '_' + str(n_file) + '.csv')
    save_next = os.path.join(next_path, 'continuous_' + chrom + '_' + str(n_file) + '.csv')
    
    
    windows_per_chr = pd.read_csv(this_windows_path, index_col=0)
    windows_per_chr = windows_per_chr.sort_values(by=['position']).reset_index(drop=True)
    
    reads_path = '/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr'
    filtered_reads_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr'
    
    this_read_path_pos = os.path.join(reads_path, chrom + '+.csv')
    reads_per_chr_pos = pd.read_csv(this_read_path_pos, index_col=0)
    
    this_filtered_reads_path_pos = os.path.join(filtered_reads_path, chrom + '+.bed')
    filtered_reads_per_chr_pos = pd.read_csv(this_filtered_reads_path_pos, names=['chr', 'start', 'end', 'read_id', 'score', 'strand'], sep='\t')
    
    filtered_reads_per_chr_pos = filtered_reads_per_chr_pos[filtered_reads_per_chr_pos['score'] >= 20].reset_index(drop=True)
    final_filtered_reads_per_chr_pos = pd.merge(reads_per_chr_pos, filtered_reads_per_chr_pos, how='inner', on='read_id')
    final_filtered_reads_per_chr_pos = final_filtered_reads_per_chr_pos.drop(columns=['start', 'end', 'score'])
    
    this_read_path_neg = os.path.join(reads_path, chrom + '-.csv')
    reads_per_chr_neg = pd.read_csv(this_read_path_neg, index_col=0)
    
    this_filtered_reads_path_neg = os.path.join(filtered_reads_path, chrom + '-.bed')
    filtered_reads_per_chr_neg = pd.read_csv(this_filtered_reads_path_neg, names=['chr', 'start', 'end', 'read_id', 'score', 'strand'], sep='\t')
    
    filtered_reads_per_chr_neg = filtered_reads_per_chr_neg[filtered_reads_per_chr_neg['score'] >= 20].reset_index(drop=True)
    final_filtered_reads_per_chr_neg = pd.merge(reads_per_chr_neg, filtered_reads_per_chr_neg, how='inner', on='read_id')
    final_filtered_reads_per_chr_neg = final_filtered_reads_per_chr_neg.drop(columns=['start', 'end', 'score'])

    positions = sorted(list(windows_per_chr['position']))
    chunks = list(chunk_it(positions, query_len))
    inputs = [[sorted(chu)[0], sorted(chu)[-1], '', chrom, '', 'same', '', final_filtered_reads_per_chr_pos,
           final_filtered_reads_per_chr_neg, this_trans_location_path_pos, this_trans_location_path_neg] for chu in chunks]
    windows_per_chr = windows_per_chr.set_index('position', drop=False)
    
    for inpu in inputs:
        forward_ipd, reverse_ipd = create_windows_per_interval_filtered(inpu)
        start_idx = inpu[0]
        end_idx = inpu[1]
        windows_per_chr.loc[start_idx: end_idx, 'forward'] = forward_ipd
        windows_per_chr.loc[start_idx: end_idx, 'reverse'] = reverse_ipd
    windows_per_chr = windows_per_chr.reset_index(drop=True)
    windows_per_chr.to_csv(save_next)
    print(save_next, 'saved.')


def create_windows_per_interval_filtered(inp):
    start, end, motif_seq, chrom, strand, direction, label, filtered_reads_per_chr_pos, filtered_reads_per_chr_neg, trans_path_pos, trans_path_neg = inp
    reads_per_win_pos = filtered_reads_per_chr_pos[(filtered_reads_per_chr_pos['mapped_start'] <= start) & (filtered_reads_per_chr_pos['mapped_end'] >= end)].reset_index(drop=True)
    reads_per_win_neg = filtered_reads_per_chr_neg[(filtered_reads_per_chr_neg['mapped_start'] <= start) & (filtered_reads_per_chr_neg['mapped_end'] >= end)].reset_index(drop=True)
    n_reads = len(reads_per_win_pos) + len(reads_per_win_neg)
    # print('#reads:', n_reads)
    switch_direction = {'same': 'opposite', 'opposite': 'same'}
    other_dir = switch_direction[direction]
    # window_name = label + '_' + chrom + strand + '_' + direction + '_' + str(start) + '_' + str(end)
    # window_name = label + '_' + chrom + '_' + str(start) + '_' + str(end)
    windows_length = end - start + 1
    window = {'chr': chrom, 'start': start, 'end': end, 'label': label, 'motif': motif_seq,
              'reads': [{'id': 0, 'direction': '', 'strand': '', 'raw': 0, 'current': 0, 'translocation': 0, 'seq': 0} for _ in range(n_reads)]}
    counter = 0
    for rd_id in range(len(reads_per_win_pos)):
        pos_mapped_start, pos_mapped_end, read_id, fast5_path, _, _ = reads_per_win_pos.loc[rd_id, :]
        trans_loc_path = os.path.join(trans_path_pos, read_id + '.npy')
        # if 0-based (indices start from 0): start inclusive:
        start_idx = start - pos_mapped_start
        end_idx = end - pos_mapped_start + 1
        if not os.path.exists(trans_loc_path):
            trans_loc_signal = compute_translocation_time_tombo(fast5_path)
            np.save(os.path.join(trans_path_pos, read_id), trans_loc_signal)
        else:
            trans_loc_signal = np.load(trans_loc_path)
        # current_signal = compute_current_tombo(fast5_path)
        # seq = compute_seq_tombo(fast5_path)
        win_trans_loc = trans_loc_signal[start_idx: end_idx]
        # win_current = current_signal[start_idx: end_idx]
        # win_seq = seq[start_idx: end_idx]
        # t10 = time.perf_counter()
        # win_raw = compute_raw_signal(fast5_path, start_idx, 100)
        # print('   pos reads, compute raw signal', time.perf_counter() - t10)
        window['reads'][counter]['id'] = read_id
        window['reads'][counter]['direction'] = direction
        window['reads'][counter]['strand'] = '+'
        # window['reads'][counter]['raw'] = win_raw
        # window['reads'][counter]['current'] = win_current
        window['reads'][counter]['translocation'] = win_trans_loc
        # window['reads'][counter]['seq'] = win_seq
        counter += 1
    for rd_id in range(len(reads_per_win_neg)):
        neg_mapped_start, neg_mapped_end, read_id, fast5_path, _, _ = reads_per_win_neg.loc[rd_id, :]
        # if 0-based (indices start from 0): start inclusive:
        start_idx = neg_mapped_end - end - 1
        end_idx = neg_mapped_end - start
        trans_loc_path = os.path.join(trans_path_neg, read_id + '.npy')
        if not os.path.exists(trans_loc_path):
            trans_loc_signal = compute_translocation_time_tombo(fast5_path)
            np.save(os.path.join(trans_path_neg, read_id), trans_loc_signal)
        else:
            trans_loc_signal = np.load(trans_loc_path)
        # current_signal = compute_current_tombo(fast5_path)
        # seq = compute_seq_tombo(fast5_path)
        win_trans_loc = trans_loc_signal[start_idx: end_idx]
        # win_current = current_signal[start_idx: end_idx]
        # win_seq = seq[start_idx: start_idx + end_idx]
        # t23 = time.perf_counter()
        # win_raw = compute_raw_signal(fast5_path, start_idx, 100)
        # print('   neg reads, compute raw signal', time.perf_counter() - t23)
        # t24 = time.perf_counter()
        window['reads'][counter]['id'] = read_id
        window['reads'][counter]['direction'] = other_dir
        window['reads'][counter]['strand'] = '-'
        # window['reads'][counter]['raw'] = win_raw
        # window['reads'][counter]['current'] = win_current
        window['reads'][counter]['translocation'] = win_trans_loc
        # window['reads'][counter]['seq'] = win_seq
        # print('   neg reads, write on list', time.perf_counter() - t24)
        counter += 1
        # print('   neg reads:', time.perf_counter() - t6)
    same_ipd = np.empty((windows_length))
    same_ipd[:] = np.nan
    opposite_ipd = np.empty((windows_length))
    opposite_ipd[:] = np.nan
    same_reads = [rd for rd in window['reads'] if rd['direction'] == 'same']
    opposite_reads = [rd for rd in window['reads'] if rd['direction'] == 'opposite']
    same_signals_list = [sr['translocation'] for sr in same_reads if len(sr['translocation']) == windows_length]
    opposite_signals_list = [sr['translocation'] for sr in opposite_reads if len(sr['translocation']) == windows_length]
    if len(same_signals_list) > 2 and len(opposite_signals_list) > 2:
        same_ipd = np.mean(np.stack(same_signals_list), axis=0)
        opposite_ipd = np.mean(np.stack(opposite_signals_list), axis=0)
    return same_ipd, opposite_ipd
    

def create_windows_per_interval(inp):
    
    # t4 = time.perf_counter()
    start, end, motif_seq, chrom, strand, direction, label, filtered_reads_per_chr_pos, filtered_reads_per_chr_neg, trans_path_pos, trans_path_neg = inp

    reads_per_win_pos = filtered_reads_per_chr_pos[(filtered_reads_per_chr_pos['mapped_start'] <= start) & (filtered_reads_per_chr_pos['mapped_end'] >= end)].reset_index(drop=True)
    reads_per_win_neg = filtered_reads_per_chr_neg[(filtered_reads_per_chr_neg['mapped_start'] <= start) & (filtered_reads_per_chr_neg['mapped_end'] >= end)].reset_index(drop=True)
    n_reads = len(reads_per_win_pos) + len(reads_per_win_neg)

    switch_direction = {'same': 'opposite', 'opposite': 'same'}
    other_dir = switch_direction[direction]
    
    # window_name = label + '_' + chrom + strand + '_' + direction + '_' + str(start) + '_' + str(end)
    window_name = label + '_' + chrom + '_' + str(start) + '_' + str(end)
    window = {'chr': chrom, 'start': start, 'end': end, 'label': label, 'motif': motif_seq,
              'reads': [{'id': 0, 'direction': '', 'strand': '', 'raw': 0, 'current': 0, 'translocation': 0, 'seq': 0} for i in range(n_reads)]}
    counter = 0
    # print('stage 1:', time.perf_counter() - t4)
    
    for rd_id in range(len(reads_per_win_pos)):
        pos_mapped_start, pos_mapped_end, read_id, fast5_path, _, _ = reads_per_win_pos.loc[rd_id, :]
        trans_loc_path = os.path.join(trans_path_pos, read_id + '.npy')

        # if 0-based (indices start from 0): start inclusive:
        start_idx = start - pos_mapped_start
        end_idx = end - pos_mapped_start + 1

        if not os.path.exists(trans_loc_path):
            trans_loc_signal = compute_translocation_time_tombo(fast5_path)
            np.save(os.path.join(trans_path_pos, read_id), trans_loc_signal)
        else:
            trans_loc_signal = np.load(trans_loc_path)
        
        current_signal = compute_current_tombo(fast5_path)
        seq = compute_seq_tombo(fast5_path)

        win_trans_loc = trans_loc_signal[start_idx: end_idx]
        win_current = current_signal[start_idx: end_idx]
        win_seq = seq[start_idx: end_idx]
        
        # t10 = time.perf_counter()
        # win_raw = compute_raw_signal(fast5_path, start_idx, 100)
        # print('   pos reads, compute raw signal', time.perf_counter() - t10)
        
        # t11 = time.perf_counter()
        window['reads'][counter]['id'] = read_id
        window['reads'][counter]['direction'] = direction
        window['reads'][counter]['strand'] = '+'
        # window['reads'][counter]['raw'] = win_raw
        window['reads'][counter]['current'] = win_current
        window['reads'][counter]['translocation'] = win_trans_loc
        window['reads'][counter]['seq'] = win_seq
        # print('   pos reads, write on list', time.perf_counter() - t11)
        counter += 1
        
        # print('   pos reads:', time.perf_counter() - t5)
        
    for rd_id in range(len(reads_per_win_neg)):
        neg_mapped_start, neg_mapped_end, read_id, fast5_path, _, _ = reads_per_win_neg.loc[rd_id, :]
        # if 0-based (indices start from 0): start inclusive:
        start_idx = neg_mapped_end - end - 1
        end_idx = neg_mapped_end - start

        trans_loc_path = os.path.join(trans_path_neg, read_id + '.npy')
        if not os.path.exists(trans_loc_path):
            trans_loc_signal = compute_translocation_time_tombo(fast5_path)
            np.save(os.path.join(trans_path_neg, read_id), trans_loc_signal)
        else:
            trans_loc_signal = np.load(trans_loc_path)

        # t21 = time.perf_counter()
        current_signal = compute_current_tombo(fast5_path)
        # print('   neg reads, compute current signal', time.perf_counter() - t21)
        seq = compute_seq_tombo(fast5_path)

        win_trans_loc = trans_loc_signal[start_idx: end_idx]
        win_current = current_signal[start_idx: end_idx]
        win_seq = seq[start_idx: start_idx + end_idx]

        # t23 = time.perf_counter()
        # win_raw = compute_raw_signal(fast5_path, start_idx, 100)
        # print('   neg reads, compute raw signal', time.perf_counter() - t23)
        
        # t24 = time.perf_counter()
        window['reads'][counter]['id'] = read_id
        window['reads'][counter]['direction'] = other_dir
        window['reads'][counter]['strand'] = '-'
        # window['reads'][counter]['raw'] = win_raw
        window['reads'][counter]['current'] = win_current
        window['reads'][counter]['translocation'] = win_trans_loc
        window['reads'][counter]['seq'] = win_seq
        # print('   neg reads, write on list', time.perf_counter() - t24)
        counter += 1
        # print('   neg reads:', time.perf_counter() - t6)
    return window, window_name


def create_windows_per_long_intervl(inp):
    
    # t4 = time.perf_counter()
    start, end, motif_seq, chrom, strand, direction, label, filtered_reads_per_chr_pos, filtered_reads_per_chr_neg, trans_path_pos, trans_path_neg = inp
    
    reads_per_win_pos = filtered_reads_per_chr_pos[(filtered_reads_per_chr_pos['mapped_start'] <= start) & (filtered_reads_per_chr_pos['mapped_end'] >= end)].reset_index(drop=True)
    reads_per_win_neg = filtered_reads_per_chr_neg[(filtered_reads_per_chr_neg['mapped_start'] <= start) & (filtered_reads_per_chr_neg['mapped_end'] >= end)].reset_index(drop=True)
    n_reads = len(reads_per_win_pos) + len(reads_per_win_neg)
    
    switch_direction = {'same': 'opposite', 'opposite': 'same'}
    other_dir = switch_direction[direction]
    
    # window_name = label + '_' + chrom + strand + '_' + direction + '_' + str(start) + '_' + str(end)
    window_name = label + '_' + chrom + '_' + str(start) + '_' + str(end)
    window = {'chr': chrom, 'start': start, 'end': end, 'label': label, 'motif': motif_seq,
              'reads': [{'id': 0, 'direction': '', 'strand': '', 'raw': 0, 'current': 0, 'translocation': 0, 'seq': 0} for i in range(n_reads)]}
    counter = 0
    # print('stage 1:', time.perf_counter() - t4)
    
    for rd_id in range(len(reads_per_win_pos)):
        # t5 = time.perf_counter()
        mapped_start, mapped_end, read_id, fast5_path, _, _ = reads_per_win_pos.loc[rd_id, :]
        trans_loc_path = os.path.join(trans_path_pos, read_id + '.npy')
        
        # t7 = time.perf_counter()
        if not os.path.exists(trans_loc_path):
            trans_loc_signal = compute_translocation_time_tombo(fast5_path)
            np.save(os.path.join(trans_path_pos, read_id), trans_loc_signal)
        else:
            trans_loc_signal = np.load(trans_loc_path)
        # print('   pos reads, compute translocation signal', time.perf_counter() - t7)
        
        # t8 = time.perf_counter()
        current_signal = compute_current_tombo(fast5_path)
        # print('   pos reads, compute current signal', time.perf_counter() - t8)
        
        # t9 = time.perf_counter()
        seq = compute_seq_tombo(fast5_path)
        # print('   pos reads, compute seq', time.perf_counter() - t9)
        
        # if 0-based (indices start from 0): start inclusive:
        start_idx = start - mapped_start
        # if not inclusive: end included
        # start_idx = start - mapped_start + 1
        win_current = current_signal[start_idx: start_idx + 100]
        win_trans_loc = trans_loc_signal[start_idx: start_idx + 100]
        win_seq = seq[start_idx: start_idx + 100]
        
        # t10 = time.perf_counter()
        # win_raw = compute_raw_signal(fast5_path, start_idx, 100)
        # print('   pos reads, compute raw signal', time.perf_counter() - t10)
        
        # t11 = time.perf_counter()
        window['reads'][counter]['id'] = read_id
        window['reads'][counter]['direction'] = direction
        window['reads'][counter]['strand'] = '+'
        # window['reads'][counter]['raw'] = win_raw
        window['reads'][counter]['current'] = win_current
        window['reads'][counter]['translocation'] = win_trans_loc
        window['reads'][counter]['seq'] = win_seq
        # print('   pos reads, write on list', time.perf_counter() - t11)
        counter += 1
        
        # print('   pos reads:', time.perf_counter() - t5)
    
    for rd_id in range(len(reads_per_win_neg)):
        # t6 = time.perf_counter()
        mapped_start, mapped_end, read_id, fast5_path, _, _ = reads_per_win_neg.loc[rd_id, :]
        # t20 = time.perf_counter()
        trans_loc_path = os.path.join(trans_path_neg, read_id + '.npy')
        if not os.path.exists(trans_loc_path):
            trans_loc_signal = compute_translocation_time_tombo(fast5_path)
            np.save(os.path.join(trans_path_neg, read_id), trans_loc_signal)
        else:
            trans_loc_signal = np.load(trans_loc_path)
        # print('   neg reads, compute translocation signal', time.perf_counter() - t20)
        
        # t21 = time.perf_counter()
        current_signal = compute_current_tombo(fast5_path)
        # print('   neg reads, compute current signal', time.perf_counter() - t21)
        
        # t22 = time.perf_counter()
        seq = compute_seq_tombo(fast5_path)
        # print('   neg reads, compute seq', time.perf_counter() - t22)
        # if 0-based (indices start from 0): start inclusive:
        start_idx = start - mapped_start
        
        # if not inclusive: end included
        # start_idx = start - mapped_start + 1
        
        win_current = current_signal[start_idx: start_idx + 100]
        win_trans_loc = trans_loc_signal[start_idx: start_idx + 100]
        win_seq = seq[start_idx: start_idx + 100]
        
        # t23 = time.perf_counter()
        # win_raw = compute_raw_signal(fast5_path, start_idx, 100)
        # print('   neg reads, compute raw signal', time.perf_counter() - t23)
        
        # t24 = time.perf_counter()
        window['reads'][counter]['id'] = read_id
        window['reads'][counter]['direction'] = other_dir
        window['reads'][counter]['strand'] = '-'
        # window['reads'][counter]['raw'] = win_raw
        window['reads'][counter]['current'] = win_current
        window['reads'][counter]['translocation'] = win_trans_loc
        window['reads'][counter]['seq'] = win_seq
        # print('   neg reads, write on list', time.perf_counter() - t24)
        counter += 1
        # print('   neg reads:', time.perf_counter() - t6)
    return window, window_name

    
def create_windows_multiple_in_1_file(inp):
    label, save_path, windows = inp
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    windows_dict = {}
    starts = []
    ends = []
    
    for win in windows:
        starts.append(win[0])
        ends.append(win[1])
        # t3 = time.perf_counter()
        window, window_name = create_windows_per_interval(win)
        # print('compute 1 window:', time.perf_counter() - t3)
        windows_dict[window_name] = window
        
    file_name = label + '_' + str(min(starts)) + '_' + str(max(ends)) + '.pkl.gz'
    # with open(os.path.join(save_path, file_name), 'wb') as handle:
    #     pickle.dump(windows_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # t4 = time.perf_counter()
    write_window_with_pickle_gzip(windows_dict, os.path.join(save_path, file_name))
    # print('write window file:', time.perf_counter() - t4)
    print('file ', os.path.join(save_path, file_name), 'saved.')


def make_input_file_job_array():
    chromosomes = ['chr' + str(i) for i in list(range(1, 23)) + ['X', 'Y']]
    # strands = ['+', '-']
    # chrstr = [ch + st for ch in chromosomes for st in strands]
    input_path = '/labs/Aguiar/non_bdna/human/scripts/creat_wins/input.txt'
    with open(input_path, 'w') as f:
        for inp in chromosomes:
            f.write(inp + '\n')
    
    sh_path = '/labs/Aguiar/non_bdna/human/scripts/creat_wins/job_array.sh'
    to_print = '#!/bin/bash\n' \
               '#BATCH --job-name=win_array\n' \
               '#SBATCH -N 1\n' \
               '#SBATCH -n 40\n' \
               '#SBATCH -c 1\n' \
               '#SBATCH --partition=general\n' \
               '#SBATCH --qos=general\n' \
               '#SBATCH --array=[0-{}]%20\n' \
               '#SBATCH --mail-type=END\n' \
               '#SBATCH --mem=10G\n' \
               '#SBATCH --mail-user=marjan.hosseini@uconn.edu\n' \
               '#SBATCH -o %x_%A_%a.out\n' \
               '#SBATCH -e %x_%A_%a.err\n' \
               '\n' \
               'echo `hostname`\n' \
               'readarray -t TaskArray < {}\n' \
               '\n' \
               'echo ${{TaskArray[${{SLURM_ARRAY_TASK_ID}}]}}\n' \
               'echo "Processing chromosome ${{TaskArray[${{SLURM_ARRAY_TASK_ID}}]}}"\n' \
               'python3 /home/FCAM/mhosseini/nonBDNA/compute_windows.py -c ${{TaskArray[${{SLURM_ARRAY_TASK_ID}}]}} -t 40\n' \
               '\n'.format(str(len(chromosomes)), input_path)
    with open(sh_path, 'w') as f:
        f.write(to_print)


def aggregate_translocation_signals_pandas():
    chrom_list = ['chr' + str(i) for i in list(range(1, 23)) + ['X', 'Y']]
    destination_folder = '/labs/Aguiar/non_bdna/annotations/windows/agg_transloc_signals/signals_df'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    
    main_older = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows/'
    
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    windows_length = 100
    columns_list = ['chr', 'strand', 'start', 'end', 'label', 'direction'] + ['val_' + str(i) for i in range(windows_length)]
    
    for chrom in chrom_list:
        for elemid in range(len(elements)):
            elem = elements[elemid]
            elem_name = elements_name[elemid]
            windows_folder = os.path.join(main_older, chrom, elem)
            files = [os.path.join(windows_folder, name) for name in os.listdir(windows_folder) if elem in name and '.pkl' in name]
            print(chrom, ',', elem_name, ', # windows:', len(files) * 100)
            this_signal = pd.DataFrame(columns=columns_list, index=range(len(files) * 200))
            counter = 0
            for file in files:
                windows = read_window_with_pickle_gzip(file)
                for interval in list(windows.keys()):
                    interval_window = windows[interval]
                    start = interval_window['start']
                    end = interval_window['end']
                    same_reads = [rd for rd in interval_window['reads'] if rd['direction'] == 'same']
                    opposite_reads = [rd for rd in interval_window['reads'] if rd['direction'] == 'opposite']
                    same_signals_list = [sr['translocation'] for sr in same_reads if len(sr['translocation']) == windows_length]
                    if len(same_signals_list) > 0:
                        same_ipd = np.mean(np.stack(same_signals_list), axis=0)
                    opposite_signals_list = [sr['translocation'] for sr in opposite_reads if len(sr['translocation']) == windows_length]
                    if len(opposite_signals_list) > 0:
                        opposite_ipd = np.mean(np.stack(opposite_signals_list), axis=0)
                    if len(same_signals_list) > 0 and len(opposite_signals_list) > 0:
                        strand = same_reads[0]['strand']
                        this_signal.loc[counter, 0:6] = chrom, strand, start, end, elem_name, 'same'
                        this_signal.loc[counter, 6:] = same_ipd
                        this_signal.loc[counter + 1, 0:6] = chrom, strand, start, end, elem_name, 'opposite'
                        this_signal.loc[counter + 1, 6:] = opposite_ipd
                        counter += 2
            this_signal = this_signal.dropna()
            this_signal.to_csv(os.path.join(destination_folder, chrom + '_' + elem + '_trans_df.csv'))
    
    all_signals = pd.DataFrame(columns=columns_list)
    dfs = [os.path.join(destination_folder, name) for name in os.listdir(destination_folder) if '.csv' in name]
    for df_path in dfs:
        df = pd.read_csv(df_path, index_col=0)
        all_signals = all_signals.append(df).reset_index(drop=True)
    save_file_name = os.path.join(destination_folder, 'translocation_df.csv')
    all_signals.to_csv(save_file_name)


def plot_non_b_types_direction_quantiles_vs_control():
    import matplotlib.pyplot as plt
    path = '/labs/Aguiar/non_bdna/annotations/windows/agg_transloc_signals/signals_df/translocation_df.csv'
    df = pd.read_csv(path, index_col=0)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA']
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    x = range(0, 100)
    direction = ['same', 'opposite']
    plot_path = '/labs/Aguiar/non_bdna/paper/plots/non_b_types_vs_control_directions'
    cols = ['val_'+str(i) for i in range(100)]
    
    df_control_dict = {dir: df[(df['label'] == 'Control') & (df['direction'] == dir)].reset_index(drop=True) for dir in direction}
    
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        for dir in direction:
            df_elem_dir = df[(df['label'] == elem_name) & (df['direction'] == dir)].reset_index(drop=True)
            df_control = df_control_dict[dir]
            signal = df_elem_dir.loc[:, cols].to_numpy()
            control_signal = df_control.loc[:, cols].to_numpy()
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for quant_id in range(len(quantiles)):
                quant = quantiles[quant_id]
                color = colors[quant_id]
                signal_quantile = np.quantile(signal, quant, axis=0)
                control_signal_quantile = np.quantile(control_signal, quant, axis=0)
                ax.plot(x, signal_quantile, color=color, label=elem_name + ' - Quantile ' + str(quant))
                ax.plot(x, control_signal_quantile, color=color, linestyle='--', label='Control - Quantile ' + str(quant))
            ax.set_xlabel('Positions in windows', fontsize=15)  # Add an x-label to the axes.
            ax.set_ylabel('Value', fontsize=15)  # Add a y-label to the axes.
            ax.legend()
            ax.set_title(elem_name + ' - ' + dir + ' strand', fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + dir + '.png'))


if __name__ == '__main__':

    if '-c' in sys.argv:
        chrom = sys.argv[sys.argv.index('-c') + 1]
    else:
        chrom = ''
        exit('Error: Select the main directory using  -c.')
    #
    # if '-s' in sys.argv:
    #     strand = sys.argv[sys.argv.index('-s') + 1]
    # else:
    #     strand = ''
    #     exit('Error: Select the main directory using  -s.')
    #
    if '-t' in sys.argv:
        threads = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        threads = 1

    # inputs = make_inputs_for_creat_windows_per_chr_strand(chrom, strand) # for making other labels windows
    # inputs = make_inputs_for_creat_windows_per_chr_strand_control(chrom, strand) # for making control windows
    
    inputs = make_inputs_for_creat_windows_per_chr(chrom)

    print('Number of inputs:', len(inputs))
    pool = Pool(threads)
    pool.map(create_windows_multiple_in_1_file, inputs)
    
    # for inp in inputs[0:3]:
    #     create_windows_multiple_in_1_file(inp)
    aggregate_translocation_signals_pandas()
    # plot_non_b_types_direction_quantiles_vs_control()

