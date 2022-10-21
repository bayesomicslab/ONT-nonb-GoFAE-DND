import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import pickle
import networkx as nx
import h5py
from utils import read_window_with_pickle_gzip
import shutil

def compute_currents(cliped_moves, cliped_current_mean):
    nz_idx = np.nonzero(cliped_moves)[0]
    windows_current = np.array_split(cliped_current_mean, nz_idx+1)[:-1]
    detected_no_seq_per_windows = cliped_moves[nz_idx]
    d2 = max(cliped_moves)
    temp_v = np.zeros((len(nz_idx), d2))
    for d in range(1, d2+1):
        d_idx = np.where(detected_no_seq_per_windows == d)
        temp_v[d_idx, 0:d] = 1
    current_per_window = np.array([np.mean(i) for i in windows_current])
    current_folded = temp_v * current_per_window[:, None]
    current_unfolded = current_folded.flatten()
    nz_final = np.nonzero(current_unfolded)
    final_current = current_unfolded[nz_final]
    
    return final_current


def make_current_signals(fast5_path):
    f = h5py.File(fast5_path, 'r')
    
    events = list(f['Analyses/Basecall_1D_000/BaseCalled_template/Events'])
    move_table = [ev[5] for ev in events]
    cliped_moves = np.array(move_table[1:])
    cliped_current_mean = [ev[0] for ev in events[1:]]
    cliped_current_std = [ev[2] for ev in events[1:]]
    base_prob = [ev[6] for ev in events[1:]]
    weights = [ev[7] for ev in events[1:]]
    
    final_current_mean = compute_currents(cliped_moves, cliped_current_mean)
    final_current_std = compute_currents(cliped_moves, cliped_current_std)
    final_prob = compute_currents(cliped_moves, base_prob)
    final_weights = compute_currents(cliped_moves, weights)
    return final_current_mean, final_current_std, final_prob, final_weights


def make_IWT_data():
    save_folder = '/labs/Aguiar/non_bdna/windows/IWT_data/'
    main_older = '/labs/Aguiar/non_bdna/windows/'
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Short_Tandem_Repeat',
                'Z_DNA_Motif', 'Controls']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Short Tandem Repeat',
                     'Z DNA', 'Controls']
    windows_length = 100
    
    datasets_df = pd.DataFrame(columns=['id', 'name', 'RegionFile'], index=range(len(elements)))
    features_datasetsTable = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsTable.loc[0, :] = 'Duration_Signal', 'Duration Signal'

    features_datasetsBED = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsBED.loc[0, :] = 'Duration_Signal', 'Duration Signal'
    
    elem_counter = 0
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    for elemid in range(len(elements)):
        elem = elements[elemid]
        elem_name = elements_name[elemid]
        print(elem)
        files = [os.path.join(main_older, name) for name in os.listdir(main_older) if elem in name and '.pickle' in name]
        feature_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        feature_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(len(files)*1000))
        counter = 0
        for file in files:
            print('processing ... :', file)
            with open(file, 'rb') as f:
                windows = pickle.load(f)
            for interval in list(windows.keys()):
                interval_window = windows[interval]
                if len(interval_window.keys()) >= 3:
                    for rd in list(interval_window.keys()):
                        if 'genome_start_position' in interval_window[rd].keys():
                            start_position = interval_window[rd]['genome_start_position']
                            break
                    seq_list, ipds_list = [], []
                    for read_id in list(interval_window.keys()):
                        if 'seq' in list(interval_window[read_id].keys()) and 'ipd' in list(interval_window[read_id].keys()):
                            if len(interval_window[read_id]['seq']) == 100 and len(interval_window[read_id]['ipd']) == 100:
                                seq_list.append(interval_window[read_id]['seq'])
                                ipds_list.append(interval_window[read_id]['ipd'])
                    if len(seq_list) > 3:
                        ipds_np = np.stack(ipds_list, axis=0)
                        final_ipds = ipds_np.mean(axis=0)
                        feature_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                        feature_elem_txt_df.loc[counter, 3:] = final_ipds
                        feature_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_ipds)
                        elem_region_df.loc[counter, :] = 'chr22', start_position, start_position + 99
                        counter += 1
        if counter > 0:
            
            feature_elem_txt_df = feature_elem_txt_df.dropna()
            feature_elem_txt_df = feature_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            feature_elem_txt_df.to_csv(os.path.join(save_folder, elem + '.txt'), sep='\t', header=None, index=0)
            
            feature_elem_bed_df = feature_elem_bed_df.dropna()
            feature_elem_bed_df = feature_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            feature_elem_bed_df.to_csv(os.path.join(save_folder, elem + '.bed'), sep='\t', header=None, index=0)
            
            elem_region_df = elem_region_df.dropna()
            elem_region_df = elem_region_df.sort_values(by='start').reset_index(drop=True)
            elem_region_df.to_csv(os.path.join(save_folder, elem + '_Regions.bed'), sep='\t', header=None, index=0)
            
            datasets_df.loc[elem_counter, :] = elem, elem_name, elem + '_Regions.bed'
            features_datasetsTable[elem] = elem + '.txt'
            features_datasetsBED[elem] = elem + '.bed'

            datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)
            features_datasetsTable.to_csv(os.path.join(save_folder, 'features_datasetsTable.txt'), sep='\t', index=0)
            features_datasetsBED.to_csv(os.path.join(save_folder, 'features_datasetsBED.txt'), sep='\t', index=0)
            elem_counter += 1

    datasets_df = datasets_df.dropna()
    datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)


def make_IWT_data_extra_features():
    features_save_folder = '/labs/Aguiar/non_bdna/features/'
    save_folder = '/labs/Aguiar/non_bdna/windows/IWT_data_extra_features/'
    main_older = '/labs/Aguiar/non_bdna/windows/final_windows'
    read_ids_index_path = '/labs/Aguiar/non_bdna/DeepMod/hx1_ab231_2/read_ids_index.csv'
    read_ids_idx_df = pd.read_csv(read_ids_index_path, index_col=0)
    features = ['Duration_Signal_', 'Current_Signal_', 'Current_Std_', 'Base_Probability_', 'Weights_']
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Short_Tandem_Repeat',
                'Z_DNA_Motif', 'Controls']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Short Tandem Repeat',
                     'Z DNA', 'Controls']
    windows_length = 100
    
    datasets_df = pd.DataFrame(columns=['id', 'name', 'RegionFile'], index=range(len(elements)))
    
    features_datasetsTable = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsTable.loc[0, :] = 'Duration_Signal', 'Duration Signal'
    features_datasetsTable.loc[1, :] = 'Current_Signal', 'Current Signal'
    features_datasetsTable.loc[2, :] = 'Current_Std', 'Current Std'
    features_datasetsTable.loc[3, :] = 'Base_Probability', 'Base Probability'
    features_datasetsTable.loc[4, :] = 'Weights', 'Weights'
    
    features_datasetsBED = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsBED.loc[0, :] = 'Duration_Signal', 'Duration Signal'
    features_datasetsBED.loc[1, :] = 'Current_Signal', 'Current Signal'
    features_datasetsBED.loc[2, :] = 'Current_Std', 'Current Std'
    features_datasetsBED.loc[3, :] = 'Base_Probability', 'Base Probability'
    features_datasetsBED.loc[4, :] = 'Weights', 'Weights'
    
    elem_counter = 0
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    for elemid in range(len(elements)):
        elem = elements[elemid]
        elem_name = elements_name[elemid]
        print(elem)
        files = [os.path.join(main_older, name) for name in os.listdir(main_older) if elem in name and '_final.pickle' in name]

        duration_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        current_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        current_std_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        prob_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        weight_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))

        duration_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        current_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        current_std_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        prob_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        weight_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        
        elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(len(files)*1000))
        
        counter = 0
        for file in files:
            print('processing ... :', file)
            with open(file, 'rb') as f:
                windows = pickle.load(f)
            for interval in list(windows.keys()):
                interval_window = windows[interval]
                if len(interval_window.keys()) >= 3:
                    for rd in list(interval_window.keys()):
                        if 'genome_start_position' in interval_window[rd].keys():
                            start_position = interval_window[rd]['genome_start_position']
                            break
                    seq_list, ipds_list, current_list, current_std_list, probs_list, weights_list = [], [], [], [], [], []
                    for read_id in list(interval_window.keys()):
                        if 'seq' in list(interval_window[read_id].keys()) and 'ipd' in list(interval_window[read_id].keys()):
                            if len(interval_window[read_id]['seq']) == 100 and len(interval_window[read_id]['ipd']) == 100:
                                seq_list.append(interval_window[read_id]['seq'])
                                ipds_list.append(interval_window[read_id]['ipd'])
                                locations_df = read_ids_idx_df[read_ids_idx_df['read_id'] == read_id]
                                fast5_file_name = locations_df.filename.values[0]
                                fast5_path = locations_df.fastq_path.values[0]
                                final_current_mean, final_current_std, final_prob, final_weights = make_current_signals(os.path.join(fast5_path, fast5_file_name))
                                
                                np.save(os.path.join(features_save_folder, 'current', fast5_file_name.split('.fast5')[0]), final_current_mean)
                                np.save(os.path.join(features_save_folder, 'current_std', fast5_file_name.split('.fast5')[0]), final_current_std)
                                np.save(os.path.join(features_save_folder, 'prob', fast5_file_name.split('.fast5')[0]), final_prob)
                                np.save(os.path.join(features_save_folder, 'weights', fast5_file_name.split('.fast5')[0]), final_weights)

                                read_start = interval_window[read_id]['read_start']

                                str_idx = start_position - read_start
                                window_current = final_current_mean[str_idx: str_idx+windows_length]
                                window_current_std = final_current_std[str_idx: str_idx+windows_length]
                                window_prob = final_prob[str_idx: str_idx+windows_length]
                                window_weights = final_weights[str_idx: str_idx+windows_length]
                                current_list.append(window_current)
                                current_std_list.append(window_current_std)
                                probs_list.append(window_prob)
                                weights_list.append(window_weights)

                    if len(seq_list) > 3:
                        ipds_np = np.stack(ipds_list, axis=0)
                        final_ipds = ipds_np.mean(axis=0)
                        
                        current_np = np.stack(current_list, axis=0)
                        final_cur = current_np.mean(axis=0)

                        current_std_np = np.stack(current_std_list, axis=0)
                        final_cur_std = current_std_np.mean(axis=0)

                        prob_np = np.stack(probs_list, axis=0)
                        final_pro = prob_np.mean(axis=0)

                        weights_np = np.stack(weights_list, axis=0)
                        final_wei = weights_np.mean(axis=0)

                        duration_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                        duration_elem_txt_df.loc[counter, 3:] = final_ipds

                        current_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                        current_elem_txt_df.loc[counter, 3:] = final_cur

                        current_std_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                        current_std_elem_txt_df.loc[counter, 3:] = final_cur_std

                        prob_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                        prob_elem_txt_df.loc[counter, 3:] = final_pro

                        weight_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                        weight_elem_txt_df.loc[counter, 3:] = final_wei

                        duration_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_ipds)
                        current_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_cur)
                        current_std_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_cur_std)
                        prob_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_pro)
                        weight_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_wei)
                        
                        elem_region_df.loc[counter, :] = 'chr22', start_position, start_position + 99
                        counter += 1
                        
        if counter > 0:
    
            duration_elem_txt_df = duration_elem_txt_df.dropna()
            current_elem_txt_df = current_elem_txt_df.dropna()
            current_std_elem_txt_df = current_std_elem_txt_df.dropna()
            prob_elem_txt_df = prob_elem_txt_df.dropna()
            weight_elem_txt_df = weight_elem_txt_df.dropna()

            duration_elem_txt_df = duration_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            current_elem_txt_df = current_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            current_std_elem_txt_df = current_std_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            prob_elem_txt_df = prob_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            weight_elem_txt_df = weight_elem_txt_df.sort_values(by='start').reset_index(drop=True)

            duration_elem_txt_df.to_csv(os.path.join(save_folder, 'Duration_Signal_' + elem + '.txt'), sep='\t', header=None, index=0)
            current_elem_txt_df.to_csv(os.path.join(save_folder, 'Current_Signal_' + elem + '.txt'), sep='\t', header=None, index=0)
            current_std_elem_txt_df.to_csv(os.path.join(save_folder, 'Current_Std_' + elem + '.txt'), sep='\t', header=None, index=0)
            prob_elem_txt_df.to_csv(os.path.join(save_folder, 'Base_Probability_' + elem + '.txt'), sep='\t', header=None, index=0)
            weight_elem_txt_df.to_csv(os.path.join(save_folder, 'Weights_' + elem + '.txt'), sep='\t', header=None, index=0)

            duration_elem_bed_df = duration_elem_bed_df.dropna()
            current_elem_bed_df = current_elem_bed_df.dropna()
            current_std_elem_bed_df = current_std_elem_bed_df.dropna()
            prob_elem_bed_df = prob_elem_bed_df.dropna()
            weight_elem_bed_df = weight_elem_bed_df.dropna()

            duration_elem_bed_df = duration_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            current_elem_bed_df = current_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            current_std_elem_bed_df = current_std_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            prob_elem_bed_df = prob_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            weight_elem_bed_df = weight_elem_bed_df.sort_values(by='start').reset_index(drop=True)

            duration_elem_bed_df.to_csv(os.path.join(save_folder, 'Duration_Signal_' + elem + '.bed'), sep='\t', header=None, index=0)
            current_elem_bed_df.to_csv(os.path.join(save_folder, 'Current_Signal_' + elem + '.bed'), sep='\t', header=None, index=0)
            current_std_elem_bed_df.to_csv(os.path.join(save_folder, 'Current_Std_' + elem + '.bed'), sep='\t', header=None, index=0)
            prob_elem_bed_df.to_csv(os.path.join(save_folder, 'Base_Probability_' + elem + '.bed'), sep='\t', header=None, index=0)
            weight_elem_bed_df.to_csv(os.path.join(save_folder, 'Weights_' + elem + '.bed'), sep='\t', header=None, index=0)
            
            elem_region_df = elem_region_df.dropna()
            elem_region_df = elem_region_df.sort_values(by='start').reset_index(drop=True)
            elem_region_df.to_csv(os.path.join(save_folder, elem + '_Regions.bed'), sep='\t', header=None, index=0)
            
            datasets_df.loc[elem_counter, :] = elem, elem_name, elem + '_Regions.bed'

            features_datasetsTable[elem] = [feat + elem + '.txt' for feat in features]
            features_datasetsBED[elem] = [feat + elem + '.bed' for feat in features]
            
            datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)
            features_datasetsTable.to_csv(os.path.join(save_folder, 'features_datasetsTable.txt'), sep='\t', index=0)
            features_datasetsBED.to_csv(os.path.join(save_folder, 'features_datasetsBED.txt'), sep='\t', index=0)
            elem_counter += 1
    
    datasets_df = datasets_df.dropna()
    datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)


def make_IWT_data_extra_features_ready():
    # features_save_folder = '/labs/Aguiar/non_bdna/features/'
    save_folder = '/labs/Aguiar/non_bdna/windows/IWT_data_extra_features/'
    main_older = '/labs/Aguiar/non_bdna/windows/final_windows'
    read_ids_index_path = '/labs/Aguiar/non_bdna/DeepMod/hx1_ab231_2/read_ids_index.csv'
    read_ids_idx_df = pd.read_csv(read_ids_index_path, index_col=0)
    features = ['Duration_Signal_', 'Current_Signal_', 'Current_Std_', 'Base_Probability_']
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Controls']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Controls']
    windows_length = 100
    datasets_df = pd.DataFrame(columns=['id', 'name', 'RegionFile'], index=range(len(elements)))
    
    features_datasetsTable = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsTable.loc[0, :] = 'Duration_Signal', 'Duration Signal'
    features_datasetsTable.loc[1, :] = 'Current_Signal', 'Current Signal'
    features_datasetsTable.loc[2, :] = 'Current_Std', 'Current Std'
    features_datasetsTable.loc[3, :] = 'Base_Probability', 'Base Probability'
    # features_datasetsTable.loc[4, :] = 'Weights', 'Weights'
    
    features_datasetsBED = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsBED.loc[0, :] = 'Duration_Signal', 'Duration Signal'
    features_datasetsBED.loc[1, :] = 'Current_Signal', 'Current Signal'
    features_datasetsBED.loc[2, :] = 'Current_Std', 'Current Std'
    features_datasetsBED.loc[3, :] = 'Base_Probability', 'Base Probability'
    # features_datasetsBED.loc[4, :] = 'Weights', 'Weights'
    
    elem_counter = 0
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    for elemid in range(len(elements)):
        elem = elements[elemid]
        elem_name = elements_name[elemid]
        print(elem)
        files = [os.path.join(main_older, name) for name in os.listdir(main_older) if elem in name and '_final.pickle' in name]
        
        duration_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        current_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        current_std_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        prob_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        # weight_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        
        duration_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        current_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        current_std_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        prob_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        weight_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        
        elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(len(files)*1000))
        
        counter = 0
        for file in files:
            print('processing ... :', file)
            with open(file, 'rb') as f:
                windows = pickle.load(f)
            for interval in list(windows.keys()):
                interval_window = windows[interval]
                start_position = interval_window['start_position']
                ipd = interval_window['ipd']
                current = interval_window['current']
                current_std = interval_window['current_std']
                probs = interval_window['probs']
                
                duration_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                duration_elem_txt_df.loc[counter, 3:] = ipd
                
                current_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                current_elem_txt_df.loc[counter, 3:] = current
                
                current_std_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                current_std_elem_txt_df.loc[counter, 3:] = current_std

                prob_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                prob_elem_txt_df.loc[counter, 3:] = probs
                
                # weight_elem_txt_df.loc[counter, 0:3] = 'chr22', start_position, start_position + 99
                # weight_elem_txt_df.loc[counter, 3:] = final_wei
                
                duration_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(ipd)
                current_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(current)
                current_std_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(current_std)
                prob_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(probs)
                # weight_elem_bed_df.loc[counter, :] = 'chr22', start_position, start_position + 99, np.mean(final_wei)
                        
                elem_region_df.loc[counter, :] = 'chr22', start_position, start_position + 99
                counter += 1
        print('counter for elem', elem , counter)
        if counter > 0:
            duration_elem_txt_df = duration_elem_txt_df.dropna()
            current_elem_txt_df = current_elem_txt_df.dropna()
            current_std_elem_txt_df = current_std_elem_txt_df.dropna()
            prob_elem_txt_df = prob_elem_txt_df.dropna()
            # weight_elem_txt_df = weight_elem_txt_df.dropna()
            
            duration_elem_txt_df = duration_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            current_elem_txt_df = current_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            current_std_elem_txt_df = current_std_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            prob_elem_txt_df = prob_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            # weight_elem_txt_df = weight_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            
            duration_elem_txt_df.to_csv(os.path.join(save_folder, 'Duration_Signal_' + elem + '.txt'), sep='\t', header=None, index=0)
            current_elem_txt_df.to_csv(os.path.join(save_folder, 'Current_Signal_' + elem + '.txt'), sep='\t', header=None, index=0)
            current_std_elem_txt_df.to_csv(os.path.join(save_folder, 'Current_Std_' + elem + '.txt'), sep='\t', header=None, index=0)
            prob_elem_txt_df.to_csv(os.path.join(save_folder, 'Base_Probability_' + elem + '.txt'), sep='\t', header=None, index=0)
            # weight_elem_txt_df.to_csv(os.path.join(save_folder, 'Weights_' + elem + '.txt'), sep='\t', header=None, index=0)
            
            duration_elem_bed_df = duration_elem_bed_df.dropna()
            current_elem_bed_df = current_elem_bed_df.dropna()
            current_std_elem_bed_df = current_std_elem_bed_df.dropna()
            prob_elem_bed_df = prob_elem_bed_df.dropna()
            # weight_elem_bed_df = weight_elem_bed_df.dropna()
            
            duration_elem_bed_df = duration_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            current_elem_bed_df = current_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            current_std_elem_bed_df = current_std_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            prob_elem_bed_df = prob_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            # weight_elem_bed_df = weight_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            
            duration_elem_bed_df.to_csv(os.path.join(save_folder, 'Duration_Signal_' + elem + '.bed'), sep='\t', header=None, index=0)
            current_elem_bed_df.to_csv(os.path.join(save_folder, 'Current_Signal_' + elem + '.bed'), sep='\t', header=None, index=0)
            current_std_elem_bed_df.to_csv(os.path.join(save_folder, 'Current_Std_' + elem + '.bed'), sep='\t', header=None, index=0)
            prob_elem_bed_df.to_csv(os.path.join(save_folder, 'Base_Probability_' + elem + '.bed'), sep='\t', header=None, index=0)
            # weight_elem_bed_df.to_csv(os.path.join(save_folder, 'Weights_' + elem + '.bed'), sep='\t', header=None, index=0)
            
            elem_region_df = elem_region_df.dropna()
            elem_region_df = elem_region_df.sort_values(by='start').reset_index(drop=True)
            elem_region_df.to_csv(os.path.join(save_folder, elem + '_Regions.bed'), sep='\t', header=None, index=0)
            
            datasets_df.loc[elem_counter, :] = elem, elem_name, elem + '_Regions.bed'
            
            features_datasetsTable[elem] = [feat + elem + '.txt' for feat in features]
            features_datasetsBED[elem] = [feat + elem + '.bed' for feat in features]
            
            datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)
            features_datasetsTable.to_csv(os.path.join(save_folder, 'features_datasetsTable.txt'), sep='\t', index=0)
            features_datasetsBED.to_csv(os.path.join(save_folder, 'features_datasetsBED.txt'), sep='\t', index=0)
            elem_counter += 1
    
    datasets_df = datasets_df.dropna()
    datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)


def remove_overlapping_windows():
    # main_older = 'IWT_data/files'
    main_older = '/labs/Aguiar/non_bdna/windows/IWT_data/'
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Short_Tandem_Repeat',
                'Z_DNA_Motif', 'Controls']
    windows_interval_graph_path = '/labs/Aguiar/non_bdna/windows/IWT_data/'
    windows_length = 100
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]

    overal_df = pd.DataFrame(columns=['start', 'end'])
    
    for elem_id in range(len(elements)):
        elem = elements[elem_id]
        elem_path = os.path.join(main_older, elem + '.bed')
        if os.path.exists(elem_path):
            this_elem = pd.read_csv(elem_path, sep='\t', names=['chr', 'start', 'end', 'value'], index_col=False)
            this_elem = this_elem.drop(columns=['chr', 'value'])
            overal_df = overal_df.append(this_elem, ignore_index=True)
            overal_df = overal_df.reset_index(drop=True)

    overal_df = overal_df.sort_values(by='start').reset_index(drop=True)
    overal_df['interval'] = list(zip(overal_df.start, overal_df.end))
    intervals = list(overal_df['interval'])
    overlapped_nodes = []
    if not os.path.exists(os.path.join(windows_interval_graph_path + 'windows_interval_graph.gexf')):
        interval_graph_chr22 = nx.interval_graph(intervals)
        edges = list(interval_graph_chr22.edges())
        overlapped_nodes = [ed[0] for ed in edges] + [ed[1] for ed in edges]
        overlapped_nodes = list(set(overlapped_nodes))
        nx.write_gexf(interval_graph_chr22, os.path.join(windows_interval_graph_path + 'windows_interval_graph.gexf'))
        
    # else:
    #     interval_graph_chr22 = nx.read_gexf(os.path.join(windows_interval_graph_path + 'windows_interval_graph.gexf'))

    for elem_id in range(len(elements)):
        elem = elements[elem_id]
        elem_path = os.path.join(main_older, elem + '.bed')
        
        if os.path.exists(elem_path):
            this_elem_bed = pd.read_csv(elem_path, sep='\t', names=['chr', 'start', 'end', 'value'], index_col=False)
            this_elem_txt = pd.read_csv(os.path.join(main_older, elem + '.txt'), sep='\t', names=['chr', 'start', 'end'] + val_cols, index_col=False)
            this_elem_region = pd.read_csv(os.path.join(main_older, elem + '_Regions.bed'), sep='\t', names=['chr', 'start', 'end'], index_col=False)
            
            this_elem_bed['interval'] = list(zip(this_elem_bed.start, this_elem_bed.end))
            this_elem_txt['interval'] = list(zip(this_elem_txt.start, this_elem_txt.end))
            this_elem_region['interval'] = list(zip(this_elem_region.start, this_elem_region.end))

            this_elem_bed = this_elem_bed[~this_elem_bed['interval'].isin(overlapped_nodes)].reset_index(drop=True)
            this_elem_txt = this_elem_txt[~this_elem_txt['interval'].isin(overlapped_nodes)].reset_index(drop=True)
            this_elem_region = this_elem_region[~this_elem_region['interval'].isin(overlapped_nodes)].reset_index(drop=True)

            this_elem_bed = this_elem_bed.drop(columns=['interval'])
            this_elem_txt = this_elem_txt.drop(columns=['interval'])
            this_elem_region = this_elem_region.drop(columns=['interval'])

            this_elem_bed.to_csv(os.path.join(main_older, elem + '.bed'), sep='\t', header=None, index=0)
            this_elem_txt.to_csv(os.path.join(main_older, elem + '.txt'), sep='\t', header=None, index=0)
            this_elem_region.to_csv(os.path.join(main_older, elem + '_Regions.bed'), sep='\t', header=None, index=0)


def make_IWT_data_extra_features_ready_paper(chrom):
    # chrom = 'chr1+'
    # features_save_folder = '/labs/Aguiar/non_bdna/features/'
    destination_folder = '/labs/Aguiar/non_bdna/annotations/windows/IWT_data_extra_features/'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    main_older = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows/'
    windows_folder = os.path.join(main_older, chrom)
    save_folder = os.path.join(destination_folder, chrom)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    features = ['Translocation_Signal_', 'Current_Signal_']
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    windows_length = 100
    datasets_df = pd.DataFrame(columns=['id', 'name', 'RegionFile'], index=range(len(elements)))
    features_datasetsTable = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsTable.loc[0, :] = 'Translocation_Signal', 'Translocation Signal'
    features_datasetsTable.loc[1, :] = 'Current_Signal', 'Current Signal'
    features_datasetsBED = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsBED.loc[0, :] = 'Translocation_Signal', 'Translocation Signal'
    features_datasetsBED.loc[1, :] = 'Current_Signal', 'Current Signal'
    elem_counter = 0
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    for elemid in range(len(elements)):
        elem = elements[elemid]
        elem_name = elements_name[elemid]
        print(elem)
        files = [os.path.join(windows_folder, elem, name) for name in os.listdir(os.path.join(windows_folder, elem)) if elem in name and '.pkl' in name]
        duration_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        current_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        duration_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        current_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(len(files)*1000))
        counter = 0
        for file in files:
            print('processing ... :', file)
            windows = read_window_with_pickle_gzip(file)
            for interval in list(windows.keys()):
                interval_window = windows[interval]
                start_position = interval_window['start']
                reads = interval_window['reads']
                translocations_list = [reads[i]['translocation'] for i in range(len(reads)) if len(reads[i]['translocation']) == windows_length]
                current_list = [reads[i]['current'] for i in range(len(reads)) if len(reads[i]['current']) == windows_length]
                if len(translocations_list) > 0 and len(current_list) > 0:
                    ipd = np.mean(np.stack(translocations_list), axis=0)
                    current = np.mean(np.stack(current_list), axis=0)
                    duration_elem_txt_df.loc[counter, 0:3] = chrom, start_position, start_position + 99
                    duration_elem_txt_df.loc[counter, 3:] = ipd
                    current_elem_txt_df.loc[counter, 0:3] = chrom, start_position, start_position + 99
                    current_elem_txt_df.loc[counter, 3:] = current
                    duration_elem_bed_df.loc[counter, :] = chrom, start_position, start_position + 99, np.mean(ipd)
                    current_elem_bed_df.loc[counter, :] = chrom, start_position, start_position + 99, np.mean(current)
                    elem_region_df.loc[counter, :] = chrom, start_position, start_position + 99
                    counter += 1
        print('counter for elem', elem, counter)
        if counter > 0:
            duration_elem_txt_df = duration_elem_txt_df.dropna()
            current_elem_txt_df = current_elem_txt_df.dropna()
            duration_elem_txt_df = duration_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            current_elem_txt_df = current_elem_txt_df.sort_values(by='start').reset_index(drop=True)
            duration_elem_txt_df.to_csv(os.path.join(save_folder, 'Translocation_Signal_' + elem + '.txt'), sep='\t', header=None, index=0)
            current_elem_txt_df.to_csv(os.path.join(save_folder, 'Current_Signal_' + elem + '.txt'), sep='\t', header=None, index=0)
            duration_elem_bed_df = duration_elem_bed_df.dropna()
            current_elem_bed_df = current_elem_bed_df.dropna()
            duration_elem_bed_df = duration_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            current_elem_bed_df = current_elem_bed_df.sort_values(by='start').reset_index(drop=True)
            duration_elem_bed_df.to_csv(os.path.join(save_folder, 'Translocation_Signal_' + elem + '.bed'), sep='\t', header=None, index=0)
            current_elem_bed_df.to_csv(os.path.join(save_folder, 'Current_Signal_' + elem + '.bed'), sep='\t', header=None, index=0)
            elem_region_df = elem_region_df.dropna()
            elem_region_df = elem_region_df.sort_values(by='start').reset_index(drop=True)
            elem_region_df.to_csv(os.path.join(save_folder, elem + '_Regions.bed'), sep='\t', header=None, index=0)
            datasets_df.loc[elem_counter, :] = elem, elem_name, elem + '_Regions.bed'
            features_datasetsTable[elem] = [feat + elem + '.txt' for feat in features]
            features_datasetsBED[elem] = [feat + elem + '.bed' for feat in features]
            datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)
            features_datasetsTable.to_csv(os.path.join(save_folder, 'features_datasetsTable.txt'), sep='\t', index=0)
            features_datasetsBED.to_csv(os.path.join(save_folder, 'features_datasetsBED.txt'), sep='\t', index=0)
            elem_counter += 1
    datasets_df = datasets_df.dropna()
    datasets_df.to_csv(os.path.join(save_folder, 'datasets.txt'), sep='\t', index=0)


def make_iwt_paper_req5():
    # main_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows/original/complete_centered_windows'
    # # control_same_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5/original/Control_Same_100_non_overlapping.npy'
    control_same_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/original/bdna/Control_Same_100_non_overlapping.npy'
    #
    # # control_opposite_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5/original/Control_Opposite_100_non_overlapping.npy'
    control_opposite_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/original/bdna/Control_Opposite_100_non_overlapping.npy'
    #
    # # main_older = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows/'
    windows_folder = '/labs/Aguiar/non_bdna/annotations/vae_windows/aggregated_motifs_req5_mean_all_chr'
    
    destination_folder = '/labs/Aguiar/non_bdna/iwt/IWT_data_req5_mean/'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    destination_folder_features = os.path.join(destination_folder, 'files')
    if not os.path.exists(destination_folder_features):
        os.mkdir(destination_folder_features)
    
    features = ['Translocation_Signal_Forward_', 'Translocation_Signal_Reverse_']
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    windows_length = 100
    datasets_df = pd.DataFrame(columns=['id', 'name', 'RegionFile'], index=range(len(elements)))
    features_datasetsTable = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsTable.loc[0, :] = 'Translocation_Signal_Forward', 'Translocation Signal Forward'
    features_datasetsTable.loc[1, :] = 'Translocation_Signal_Reverse', 'Translocation Signal Reverse'
    features_datasetsBED = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsBED.loc[0, :] = 'Translocation_Signal_Forward', 'Translocation Signal Forward'
    features_datasetsBED.loc[1, :] = 'Translocation_Signal_Reverse', 'Translocation Signal Reverse'
    elem_counter = 0
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    for elemid in range(len(elements)):
        elem = elements[elemid]
        elem_name = elements_name[elemid]
        print(elem)
        # files = [os.path.join(windows_folder, elem, name) for name in os.listdir(os.path.join(windows_folder, elem)) if elem in name and '.pkl' in name]
        files = [os.path.join(windows_folder, name) for name in os.listdir(windows_folder) if elem in name and '.pkl' in name]
        duration_elem_bed_df_forward = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(1000000))
        duration_elem_bed_df_reverse = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(1000000))
        # current_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        duration_elem_txt_df_forward = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(1000000))
        duration_elem_txt_df_reverse = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(1000000))
        # current_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(1000000))
        if elem != 'Control':
            counter = 0
            for file in files:
                print('processing ... :', file)
                windows = read_window_with_pickle_gzip(file)
                # chrom = 'chr' + file.split('chr')[-1].split('_')[0]
                
                for interval in list(windows.keys()):
                    interval_window = windows[interval]
                    start_position = interval_window['start']
                    end_position = interval_window['end']
                    chrom = windows[0]['chr']
                    length = end_position - start_position + 1
                    start_idx = int(np.floor((length - windows_length)/2))
                    forward = interval_window['same'][start_idx: start_idx + windows_length]
                    reverse = interval_window['opposite'][start_idx: start_idx + windows_length]
                    start = start_position + start_idx

                    duration_elem_txt_df_forward.loc[counter, 0:3] = chrom, start, start + 99
                    duration_elem_txt_df_forward.loc[counter, 3:] = forward
    
                    duration_elem_txt_df_reverse.loc[counter, 0:3] = chrom, start, start + 99
                    duration_elem_txt_df_reverse.loc[counter, 3:] = reverse
    
                    duration_elem_bed_df_forward.loc[counter, :] = chrom, start, start + 99, np.mean(forward)
                    duration_elem_bed_df_reverse.loc[counter, :] = chrom, start, start + 99, np.mean(reverse)
                    
                    elem_region_df.loc[counter, :] = chrom, start, start + 99
                    counter += 1
        else:
            chrom = 'chr25'
            forward_np = np.load(control_same_path)
            reverse_np = np.load(control_opposite_path)
            mean_forward = np.mean(forward_np, axis=1)
            mean_reverse = np.mean(reverse_np, axis=1)
            n_bdna = np.min([300000, forward_np.shape[0]])
            
            duration_elem_bed_df_forward = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(n_bdna))
            duration_elem_bed_df_reverse = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(n_bdna))
            duration_elem_txt_df_forward = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(n_bdna))
            duration_elem_txt_df_reverse = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(n_bdna))
            elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(n_bdna))
            
            starts = list(range(1000, n_bdna*150 + 1000, 150))
            ends = [st + 99 for st in starts]
            
            duration_elem_txt_df_forward['chr'] = chrom
            duration_elem_txt_df_forward['start'] = starts
            duration_elem_txt_df_forward['end'] = ends
            duration_elem_txt_df_forward[val_cols] = forward_np[0:n_bdna, :]
            
            duration_elem_txt_df_reverse['chr'] = chrom
            duration_elem_txt_df_reverse['start'] = starts
            duration_elem_txt_df_reverse['end'] = ends
            duration_elem_txt_df_reverse[val_cols] = reverse_np[0:n_bdna, :]
            
            duration_elem_bed_df_reverse['chr'] = chrom
            duration_elem_bed_df_reverse['start'] = starts
            duration_elem_bed_df_reverse['end'] = ends
            duration_elem_bed_df_reverse['value'] = mean_forward[0:n_bdna]
            
            duration_elem_bed_df_reverse['chr'] = chrom
            duration_elem_bed_df_reverse['start'] = starts
            duration_elem_bed_df_reverse['end'] = ends
            duration_elem_bed_df_reverse['value'] = mean_reverse[0:n_bdna]
            
            elem_region_df['chr'] = chrom
            elem_region_df['start'] = starts
            elem_region_df['end'] = ends
            counter = n_bdna
            # for counter in range(n_bdna):
            #     start = starts[counter]
            #
            #     forward = forward_np[counter, :]
            #     reverse = reverse_np[counter, :]
            #
            #     duration_elem_txt_df_forward.loc[counter, 0:3] = chrom, start, start + 99
            #     duration_elem_txt_df_forward.loc[counter, 3:] = forward
            #
            #     duration_elem_txt_df_reverse.loc[counter, 0:3] = chrom, start, start + 99
            #     duration_elem_txt_df_reverse.loc[counter, 3:] = reverse
            #
            #     duration_elem_bed_df_forward.loc[counter, :] = chrom, start, start + 99, np.mean(forward)
            #     duration_elem_bed_df_reverse.loc[counter, :] = chrom, start, start + 99, np.mean(reverse)
            #
            #     elem_region_df.loc[counter, :] = chrom, start, start + 99
                    
        print('counter for elem', elem, counter)
        if counter > 0:
            duration_elem_txt_df_forward = duration_elem_txt_df_forward.dropna()
            duration_elem_txt_df_reverse = duration_elem_txt_df_reverse.dropna()
            duration_elem_txt_df_forward = duration_elem_txt_df_forward.sort_values(by='start').reset_index(drop=True)
            duration_elem_txt_df_reverse = duration_elem_txt_df_reverse.sort_values(by='start').reset_index(drop=True)
            duration_elem_txt_df_forward.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Forward_' + elem + '.txt'), sep='\t', header=None, index=0)
            duration_elem_txt_df_reverse.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Reverse_' + elem + '.txt'), sep='\t', header=None, index=0)
            
            duration_elem_bed_df_forward = duration_elem_bed_df_forward.dropna()
            duration_elem_bed_df_reverse = duration_elem_bed_df_reverse.dropna()
            duration_elem_bed_df_forward = duration_elem_bed_df_forward.sort_values(by='start').reset_index(drop=True)
            duration_elem_bed_df_reverse = duration_elem_bed_df_reverse.sort_values(by='start').reset_index(drop=True)
            duration_elem_bed_df_forward.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Forward_' + elem + '.bed'), sep='\t', header=None, index=0)
            duration_elem_bed_df_reverse.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Reverse_' + elem + '.bed'), sep='\t', header=None, index=0)
            
            elem_region_df = elem_region_df.dropna()
            elem_region_df = elem_region_df.sort_values(by='start').reset_index(drop=True)
            elem_region_df.to_csv(os.path.join(destination_folder_features, elem + '_Regions.bed'), sep='\t', header=None, index=0)
            datasets_df.loc[elem_counter, :] = elem, elem_name, elem + '_Regions.bed'
            features_datasetsTable[elem] = [feat + elem + '.txt' for feat in features]
            features_datasetsBED[elem] = [feat + elem + '.bed' for feat in features]
            datasets_df.to_csv(os.path.join(destination_folder, 'datasets.txt'), sep='\t', index=0)
            features_datasetsTable.to_csv(os.path.join(destination_folder, 'features_datasetsTable.txt'), sep='\t', index=0)
            features_datasetsBED.to_csv(os.path.join(destination_folder, 'features_datasetsBED.txt'), sep='\t', index=0)
            elem_counter += 1
    datasets_df = datasets_df.dropna()
    datasets_df.to_csv(os.path.join(destination_folder, 'datasets.txt'), sep='\t', index=0)


def make_iwt_paper_req5_chromosome_aware():
    max_rows = 1000000
    windows_folder = '/labs/Aguiar/non_bdna/annotations/vae_windows/aggregated_motifs_req5_mean'
    
    destination_folder = '/labs/Aguiar/non_bdna/iwt/IWT_data_req5_mean/'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    destination_folder_features = os.path.join(destination_folder, 'files')
    if not os.path.exists(destination_folder_features):
        os.mkdir(destination_folder_features)
    
    features = ['Translocation_Signal_Forward_', 'Translocation_Signal_Reverse_']
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    windows_length = 100
    datasets_df = pd.DataFrame(columns=['id', 'name', 'RegionFile'], index=range(len(elements)))
    features_datasetsTable = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsTable.loc[0, :] = 'Translocation_Signal_Forward', 'Translocation Signal Forward'
    features_datasetsTable.loc[1, :] = 'Translocation_Signal_Reverse', 'Translocation Signal Reverse'
    features_datasetsBED = pd.DataFrame(columns=['id', 'name'], index=[0])
    features_datasetsBED.loc[0, :] = 'Translocation_Signal_Forward', 'Translocation Signal Forward'
    features_datasetsBED.loc[1, :] = 'Translocation_Signal_Reverse', 'Translocation Signal Reverse'
    elem_counter = 0
    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    for elemid in range(len(elements)):
        elem = elements[elemid]
        elem_name = elements_name[elemid]
        print(elem)
        files = [os.path.join(windows_folder, name) for name in os.listdir(windows_folder) if elem in name and '.pkl' in name]
        duration_elem_bed_df_forward = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(max_rows))
        duration_elem_bed_df_reverse = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(max_rows))
        # current_elem_bed_df = pd.DataFrame(columns=['chr', 'start', 'end', 'value'], index=range(len(files)*1000))
        duration_elem_txt_df_forward = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(max_rows))
        duration_elem_txt_df_reverse = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(max_rows))
        # current_elem_txt_df = pd.DataFrame(columns=['chr', 'start', 'end'] + val_cols, index=range(len(files)*1000))
        elem_region_df = pd.DataFrame(columns=['chr', 'start', 'end'], index=range(max_rows))
        if elem != 'Control':
            counter = 0
            for file in files:
                print('processing ... :', file)
                windows = read_window_with_pickle_gzip(file)
                # chrom = 'chr' + file.split('chr')[-1].split('_')[0]
                
                for interval in list(windows.keys()):
                    interval_window = windows[interval]
                    start_position = interval_window['start']
                    end_position = interval_window['end']
                    chrom = interval_window['chr']
                    length = end_position - start_position + 1
                    start_idx = int(np.floor((length - windows_length)/2))
                    forward = interval_window['same'][start_idx: start_idx + windows_length]
                    reverse = interval_window['opposite'][start_idx: start_idx + windows_length]
                    start = start_position + start_idx
                    
                    duration_elem_txt_df_forward.loc[counter, 0:3] = chrom, start, start + 99
                    duration_elem_txt_df_forward.loc[counter, 3:] = forward
                    
                    duration_elem_txt_df_reverse.loc[counter, 0:3] = chrom, start, start + 99
                    duration_elem_txt_df_reverse.loc[counter, 3:] = reverse
                    
                    duration_elem_bed_df_forward.loc[counter, :] = chrom, start, start + 99, np.mean(forward)
                    duration_elem_bed_df_reverse.loc[counter, :] = chrom, start, start + 99, np.mean(reverse)
                    
                    elem_region_df.loc[counter, :] = chrom, start, start + 99
                    counter += 1
                    if counter == max_rows:
                        break
                if counter == max_rows:
                    break
                    
        else:
            counter = 0
            for file in files:
                print('processing ... :', file)
                windows = read_window_with_pickle_gzip(file)
                for interval in list(windows.keys()):
                    interval_window = windows[interval]
                    start_position = interval_window['start']
                    end_position = interval_window['end']
                    chrom = interval_window['chr']
                    length = end_position - start_position + 1
                    if length >= 1200:
                        
                        forward = interval_window['same'][100: -100]
                        forward_np = forward[0:1000].reshape(-1, 100)
                        reverse = interval_window['opposite'][100: -100]
                        reverse_np = reverse[0:1000].reshape(-1, 100)
                        mean_forward = np.mean(forward_np, axis=1)
                        mean_reverse = np.mean(reverse_np, axis=1)
                        starts = list(range(start_position + 100, start_position + 1100, 100))
                        ends = [st + 99 for st in starts]
            
                        duration_elem_txt_df_forward.iloc[counter:counter+10, 0] = chrom
                        duration_elem_txt_df_forward.iloc[counter:counter+10, 1] = starts
                        duration_elem_txt_df_forward.iloc[counter:counter+10, 2] = ends
                        duration_elem_txt_df_forward.iloc[counter:counter+10, 3:] = forward_np
            
                        duration_elem_txt_df_reverse.iloc[counter:counter+10, 0] = chrom
                        duration_elem_txt_df_reverse.iloc[counter:counter+10, 1] = starts
                        duration_elem_txt_df_reverse.iloc[counter:counter+10, 2] = ends
                        duration_elem_txt_df_reverse.iloc[counter:counter+10, 3:] = reverse_np
            
                        duration_elem_bed_df_forward.iloc[counter:counter+10, 0] = chrom
                        duration_elem_bed_df_forward.iloc[counter:counter+10, 1] = starts
                        duration_elem_bed_df_forward.iloc[counter:counter+10, 2] = ends
                        duration_elem_bed_df_forward.iloc[counter:counter+10, 3] = mean_forward
            
                        duration_elem_bed_df_reverse.iloc[counter:counter+10, 0] = chrom
                        duration_elem_bed_df_reverse.iloc[counter:counter+10, 1] = starts
                        duration_elem_bed_df_reverse.iloc[counter:counter+10, 2] = ends
                        duration_elem_bed_df_reverse.iloc[counter:counter+10, 3] = mean_reverse
            
                        elem_region_df.iloc[counter:counter+10, 0] = chrom
                        elem_region_df.iloc[counter:counter+10, 1] = starts
                        elem_region_df.iloc[counter:counter+10, 2] = ends
                        counter += 10
                        if counter == max_rows:
                            break
                if counter == max_rows:
                    break
                
        print('counter for elem', elem, counter)
        if counter > 0:
            duration_elem_txt_df_forward = duration_elem_txt_df_forward.dropna()
            duration_elem_txt_df_reverse = duration_elem_txt_df_reverse.dropna()
            duration_elem_txt_df_forward = duration_elem_txt_df_forward.sort_values(by='start').reset_index(drop=True)
            duration_elem_txt_df_reverse = duration_elem_txt_df_reverse.sort_values(by='start').reset_index(drop=True)
            duration_elem_txt_df_forward.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Forward_' + elem + '.txt'), sep='\t', header=None, index=0)
            duration_elem_txt_df_reverse.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Reverse_' + elem + '.txt'), sep='\t', header=None, index=0)
            
            duration_elem_bed_df_forward = duration_elem_bed_df_forward.dropna()
            duration_elem_bed_df_reverse = duration_elem_bed_df_reverse.dropna()
            duration_elem_bed_df_forward = duration_elem_bed_df_forward.sort_values(by='start').reset_index(drop=True)
            duration_elem_bed_df_reverse = duration_elem_bed_df_reverse.sort_values(by='start').reset_index(drop=True)
            duration_elem_bed_df_forward.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Forward_' + elem + '.bed'), sep='\t', header=None, index=0)
            duration_elem_bed_df_reverse.to_csv(os.path.join(destination_folder_features, 'Translocation_Signal_Reverse_' + elem + '.bed'), sep='\t', header=None, index=0)
            
            elem_region_df = elem_region_df.dropna()
            elem_region_df = elem_region_df.sort_values(by='start').reset_index(drop=True)
            elem_region_df.to_csv(os.path.join(destination_folder_features, elem + '_Regions.bed'), sep='\t', header=None, index=0)
            datasets_df.loc[elem_counter, :] = elem, elem_name, elem + '_Regions.bed'
            features_datasetsTable[elem] = [feat + elem + '.txt' for feat in features]
            features_datasetsBED[elem] = [feat + elem + '.bed' for feat in features]
            datasets_df.to_csv(os.path.join(destination_folder, 'datasets.txt'), sep='\t', index=0)
            features_datasetsTable.to_csv(os.path.join(destination_folder, 'features_datasetsTable.txt'), sep='\t', index=0)
            features_datasetsBED.to_csv(os.path.join(destination_folder, 'features_datasetsBED.txt'), sep='\t', index=0)
            elem_counter += 1
    datasets_df = datasets_df.dropna()
    datasets_df.to_csv(os.path.join(destination_folder, 'datasets.txt'), sep='\t', index=0)


def prepare_iwt_R_script():
    windows_length = 100
    main_folder = '/labs/Aguiar/non_bdna/iwt/IWT_data_req5_mean/'
    features = ['Translocation_Signal_Forward_', 'Translocation_Signal_Reverse_']
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']

    val_cols = ['val'+str(i).zfill(2) for i in range(windows_length)]
    txt_cols = ['chr', 'start', 'end'] + val_cols
    max_n_samples = 10000
    test_folder = os.path.join(main_folder, 'iwt_' + str(max_n_samples))
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    test_files_folder = os.path.join(test_folder, 'files')
    if not os.path.exists(test_files_folder):
        os.mkdir(test_files_folder)

    shutil.copyfile(os.path.join(main_folder, 'datasets.txt'), os.path.join(test_folder, 'datasets.txt'))
    shutil.copyfile(os.path.join(main_folder, 'features_datasetsBED.txt'), os.path.join(test_folder, 'features_datasetsBED.txt'))
    shutil.copyfile(os.path.join(main_folder, 'features_datasetsTable.txt'), os.path.join(test_folder, 'features_datasetsTable.txt'))
    
    for element in elements:
        region_path = os.path.join(main_folder, 'files', element + '_Regions.bed')
        region_df = pd.read_csv(region_path, sep='\t', names=['chr', 'start', 'end'])
        region_df = region_df[:max_n_samples].reset_index(drop=True)
        region_df.to_csv(os.path.join(test_files_folder, element + '_Regions.bed'), sep='\t', header=None, index=0)
        for feature in features:
            text_path = os.path.join(main_folder, 'files', feature + element + '.txt')
            text_feature_df = pd.read_csv(text_path, sep='\t', names=txt_cols)
            text_feature_df = text_feature_df[:max_n_samples].reset_index(drop=True)
            text_feature_df.to_csv(os.path.join(test_files_folder, feature + element + '.txt'), sep='\t', header=None, index=0)
            
            bed_path = os.path.join(main_folder, 'files', feature + element + '.bed')
            bed_feature_df = pd.read_csv(bed_path, sep='\t', names=['chr', 'start', 'end', 'value'])
            bed_feature_df = bed_feature_df[:max_n_samples].reset_index(drop=True)
            bed_feature_df.to_csv(os.path.join(test_files_folder, feature + element + '.bed'), sep='\t', header=None, index=0)


def make_signals_numpy(chrom):
    destination_folder = '/labs/Aguiar/non_bdna/annotations/windows/Signals_np/'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    main_older = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows/'
    windows_folder = os.path.join(main_older, chrom)
    save_folder = os.path.join(destination_folder, chrom)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    windows_length = 100
    elem_counter = 0
    for elemid in range(len(elements)):
        elem = elements[elemid]
        print(elem)
        files = [os.path.join(windows_folder, elem, name) for name in os.listdir(os.path.join(windows_folder, elem)) if elem in name and '.pkl' in name]
        sizes = (len(files)*1000, windows_length)
        duration_np = np.zeros(sizes)
        current_np = np.zeros(sizes)

        counter = 0
        for file in files:
            # print('processing ... :', file)
            windows = read_window_with_pickle_gzip(file)
            for interval in list(windows.keys()):
                interval_window = windows[interval]
                reads = interval_window['reads']
                translocations_list = [reads[i]['translocation'] for i in range(len(reads)) if len(reads[i]['translocation']) == windows_length]
                current_list = [reads[i]['current'] for i in range(len(reads)) if len(reads[i]['current']) == windows_length]
                if len(translocations_list) > 0 and len(current_list) > 0:
                    ipd = np.mean(np.stack(translocations_list), axis=0)
                    current = np.mean(np.stack(current_list), axis=0)
                    duration_np[counter, :] = ipd
                    current_np[counter, :] = current
                    counter += 1
        print('counter for elem', elem, counter)
        if counter > 0:
            duration_np = duration_np[~np.all(duration_np == 0, axis=1)]
            current_np = current_np[~np.all(current_np == 0, axis=1)]

            np.savetxt(os.path.join(save_folder, 'Translocation_' + elem + '.npy'), duration_np)
            print('Saved', os.path.join(save_folder, 'Translocation_' + elem + '.npy'))

            np.savetxt(os.path.join(save_folder, 'Current_' + elem + '.npy'), current_np)
            print('Saved', os.path.join(save_folder, 'Current_' + elem + '.npy'))
            elem_counter += 1
    

def aggregate_translocation_signals_numpy(chrom):
    
    destination_folder = '/labs/Aguiar/non_bdna/annotations/windows/agg_transloc_signals/signals_np'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    main_older = '/labs/Aguiar/non_bdna/annotations/windows/initial_windows/'
    windows_folder = os.path.join(main_older, chrom)
    save_folder = os.path.join(destination_folder, chrom)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    windows_length = 100
    elem_counter = 0
    
    for elemid in range(len(elements)):
        elem = elements[elemid]
        print(elem)
        files = [os.path.join(windows_folder, elem, name) for name in os.listdir(os.path.join(windows_folder, elem)) if elem in name and '.pkl' in name]
        sizes = (len(files)*1000, windows_length)
        duration_np = np.zeros(sizes)
        
        counter = 0
        for file in files:
            # print('processing ... :', file)
            windows = read_window_with_pickle_gzip(file)
            for interval in list(windows.keys()):
                interval_window = windows[interval]
                reads = interval_window['reads']
                translocations_list = [reads[i]['translocation'] for i in range(len(reads)) if len(reads[i]['translocation']) == windows_length]
                if len(translocations_list) > 0:
                    ipd = np.mean(np.stack(translocations_list), axis=0)
                    duration_np[counter, :] = ipd
                    counter += 1
        print('counter for elem', elem, counter)
        if counter > 0:
            duration_np = duration_np[~np.all(duration_np == 0, axis=1)]
            
            np.savetxt(os.path.join(save_folder, 'Translocation_' + elem + '.npy'), duration_np)
            print('Saved', os.path.join(save_folder, 'Translocation_' + elem + '.npy'))
            
            elem_counter += 1


def plot_elems():
    chrom_list = ['chr1+', 'chr2+', 'chr3+', 'chr4+', 'chr5+', 'chr6+', 'chr7+']
    import matplotlib.pyplot as plt
    main_path = 'D:\\UCONN\\nonBDNA\\IWT\\IWT_paper\\Signals_np'
    
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    features = ['Translocation', 'Current']
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    x = range(0, 100)

    for chrom in chrom_list:
        
        for quant in quantiles:
            fig, axis = plt.subplots(2, 1, figsize=(13, 10))
            for i in range(2):
                feature = features[i]
                for elem_id in range(len(elements)):
                    elem = elements[elem_id]
                    elem_name = elements_name[elem_id]
                    path = os.path.join(main_path, chrom, feature + '_' + elem + '.npy')
                    signal = pd.read_csv(path, sep=' ', header=None)
                    signal_quantile = np.quantile(signal, quant, axis=0)
                    # signal = np.load(path, allow_pickle=True)
                    axis[i].plot(x, signal_quantile, label=elem_name)  # Plot some data on the axes.
                axis[i].set_title(chrom + ' - ' + feature + ' - ' + 'Quantile:' + str(quant), fontsize=20)
                axis[i].set_xlabel('Positions in windows', fontsize=15)  # Add an x-label to the axes.
                axis[i].set_ylabel('Value', fontsize=15)  # Add a y-label to the axes.
                axis[i].legend()  # Add a legend.
            plt.tight_layout()
            plt.savefig(os.path.join(main_path, chrom + '_quantile_' + str(quant) + '.png'))


def plot_non_b_types():
    import matplotlib.pyplot as plt
    main_path = '/labs/Aguiar/non_bdna/annotations/windows/Signals_translocation/'
    plot_path = '/labs/Aguiar/non_bdna/paper/plots'
    chrom_list = ['chr' + str(i) for i in list(range(1, 23)) + ['X', 'Y']]
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    features = ['Translocation']
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    x = range(0, 100)
    
    elems_pd_list = {feature: [pd.DataFrame() for elem_id in range(len(elements))] for feature in features}
    
    for elem_id in range(len(elements)):
        elem = elements[elem_id]
        for i in range(len(features)):
            feature = features[i]
            for chrom in chrom_list:
                path = os.path.join(main_path, chrom, feature + '_' + elem + '.npy')
                signal = pd.read_csv(path, sep=' ', header=None)
                elems_pd_list[feature][elem_id] = elems_pd_list[feature][elem_id].append(signal).reset_index(drop=True)
        
    for quant in quantiles:
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        for elem_id in range(len(elements)):
            elem_name = elements_name[elem_id]
            for i in range(len(features)):
                feature = features[i]
                signal = elems_pd_list[feature][elem_id]
                signal_quantile = np.quantile(signal, quant, axis=0)
                ax.plot(x, signal_quantile, label=elem_name)  # Plot some data on the axes.
                ax.set_title(feature + ' - ' + 'Quantile:' + str(quant), fontsize=20)
                ax.set_xlabel('Positions in windows', fontsize=15)  # Add an x-label to the axes.
                ax.set_ylabel('Value', fontsize=15)  # Add a y-label to the axes.
                ax.legend()
        plt.tight_layout()
        
        plt.savefig(os.path.join(plot_path, 'non_b_types_quantile_' + str(quant) + '.png'))


def plot_non_b_types_direction_quantiles():
    import matplotlib.pyplot as plt
    path = '/labs/Aguiar/non_bdna/annotations/windows/agg_transloc_signals/signals_df/translocation_df.csv'
    df = pd.read_csv(path, index_col=0)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif', 'Control']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA', 'Control']
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    x = range(0, 100)
    direction = ['same', 'opposite']
    plot_path = '/labs/Aguiar/non_bdna/paper/plots'
    cols = ['val_'+str(i) for i in range(100)]
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        for dir in direction:
            df_elem_dir = df[(df['label'] == elem_name) & (df['direction'] == dir)].reset_index(drop=True)
            signal = df_elem_dir.loc[:, cols].to_numpy()
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for quant in quantiles:
                signal_quantile = np.quantile(signal, quant, axis=0)
                ax.plot(x, signal_quantile, label='Quantile ' + str(quant))  # Plot some data on the axes.
            ax.set_xlabel('Positions in windows', fontsize=15)  # Add an x-label to the axes.
            ax.set_ylabel('Value', fontsize=15)  # Add a y-label to the axes.
            ax.legend()
            ax.set_title(elem_name + ' - ' + dir + ' - ', fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, elem + '_' + dir + '.png'))


def plot_non_b_types_direction_quantiles_vs_control():
    import matplotlib.pyplot as plt
    path = '/labs/Aguiar/non_bdna/annotations/windows/agg_transloc_signals/signals_df/translocation_df.csv'
    df = pd.read_csv(path, index_col=0)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA']
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple' ]
    x = range(0, 100)
    direction = ['same', 'opposite']
    plot_path = '/labs/Aguiar/non_bdna/paper/plots'
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
            ax.set_title(elem_name + ' - ' + dir + ' - ', fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + dir + '.png'))


def plot_non_b_types_direction_quantiles_vs_control_new_data():
    import matplotlib.pyplot as plt
    # main_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/original/outliers/centered_windows'
    main_path = 'Data/windows/centered_windows'
    # main_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows/original/complete_centered_windows'
    # control_same_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/original/bdna/Control_Same_100_non_overlapping.npy'
    # control_same_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows/original/Control_Same_100_non_overlapping.npy'
    control_same_path = 'Data/windows/Control_Same_100_non_overlapping.npy'
    
    # control_opposite_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5_mean/original/bdna/Control_Opposite_100_non_overlapping.npy'
    # control_opposite_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows/original/Control_Opposite_100_non_overlapping.npy'
    control_opposite_path = 'Data/windows/Control_Opposite_100_non_overlapping.npy'
    
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_path = {nonb: os.path.join(main_path, nonb + '_centered.csv') for nonb in elements}
    elements_name = ['A Phased Repeat', 'G Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z DNA']
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple' ]
    x = range(0, 100)
    direction = ['same', 'opposite']
    direction_cols = {'same': ['forward_'+str(i) for i in range(100)], 'opposite': ['reverse_'+str(i) for i in range(100)]}
    control_signals = {'same': np.load(control_same_path), 'opposite': np.load(control_opposite_path)}
    direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
    # plot_path = '/labs/Aguiar/non_bdna/paper/plots/req5_mean'
    plot_path = 'Figures/req5_mean'
    
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        df = pd.read_csv(elements_path[elem], index_col=0)
        for dir in direction:
            cols = direction_cols[dir]
            signal = df.loc[:, cols].to_numpy()
            control_signal = control_signals[dir]
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            for quant_id in range(len(quantiles)):
                quant = quantiles[quant_id]
                color = colors[quant_id]
                signal_quantile = np.quantile(signal, quant, axis=0)
                control_signal_quantile = np.quantile(control_signal, quant, axis=0)
                ax.plot(x, signal_quantile, color=color, label=str(quant), linewidth=4)
                ax.plot(x, control_signal_quantile, color=color, linestyle='--', linewidth=3)
            # max_val = np.max(np.quantile(signal, 0.95, axis=0))
            ax.set_xlabel('Position in Window', fontsize=27)
            ax.set_ylabel('Translocation Time', fontsize=27)
            # leg = plt.legend(loc=(1.03, 0.55), title="Quantile", prop={'size': 14},
            #                  title_fontsize=15)
            # ax.add_artist(leg)
            # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
            # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
            #            title="Structure", prop={'size': 14}, title_fontsize=15)
            plt.setp(ax.get_xticklabels(), Fontsize=22)
            plt.setp(ax.get_yticklabels(), Fontsize=22)
            # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
            # ax.set_ylim(top=max_val+0.00053)
            ax.set_title(elem_name + ' ' + direction_names[dir], fontsize=20)
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, elem + '_Control_' + direction_names[dir] + '.png'))


def make_r_script(data_path):
    plot_path = os.path.join(data_path, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    rdata_path = os.path.join(data_path, 'rdata')
    if not os.path.exists(rdata_path):
        os.mkdir(rdata_path)

    regions_path = os.path.join(rdata_path, 'translocation_regions.RData')
    quantile_model_path = os.path.join(rdata_path, 'iwt_results_1000_quantile_pairs.RData')
    median_model_path = os.path.join(rdata_path, 'iwt_results_3000_median_pairs.RData')
    to_print = 'if (!require("BiocManager", quietly = TRUE))\n'\
    '    install.packages("BiocManager")\n'\
    'BiocManager::install("IWTomics")\n'\
    'library(IWTomics)\n'\
    'data_path <- '+ data_path +'\n' \
    '\n' \
    '# make data:\n' \
    'datasets=read.table(file.path(' + data_path + ',"datasets.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)\n'\
    'features_datasetsTable=read.table(file.path(' + data_path + ',"features_datasetsTable.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)\n'\
    'regionsFeatures=IWTomics::IWTomicsData(datasets$RegionFile,features_datasetsTable[,3:10],"center", datasets$id,datasets$name, features_datasetsTable$id,features_datasetsTable$name, path=file.path(' + data_path + ',"files"))\n'\
    'save(regionsFeatures,file=paste0(' + regions_path + '))\n' \
    '\n' \
    '# Quantile test:\n' \
    'quantile_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c("A_Phased_Repeat", "G_Quadruplex_Motif", "Inverted_Repeat", "Mirror_Repeat", "Direct_Repeat", "Short_Tandem_Repeat","Z_DNA_Motif"), id_region2=c("Control", "Control", "Control", "Control", "Control", "Control", "Control"), statistics="quantile",probs=c(0.05,0.25,0.5,0.75,0.95),B=1000)\n'\
    'save(quantile_regionsFeatures_test_pairs,file=paste0(' + quantile_model_path +'))\n' \
    '\n' \
    '# Quantile test:\n' \
    'mean_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c("A_Phased_Repeat", "G_Quadruplex_Motif", "Inverted_Repeat","Mirror_Repeat", "Direct_Repeat", "Short_Tandem_Repeat", "Z_DNA_Motif"), id_region2=c("Control", "Control", "Control", "Control", "Control", "Control", "Control"), statistics="median",B=1000)\n'\
    'save(mean_regionsFeatures_test_pairs,file=paste0(' + median_model_path + '))\n' \
    '\n' \
    '# Adjusted p-value for median test:\n' \
    'adjusted_pval(mean_regionsFeatures_test_pairs)\n'
    '\n' \
    '# Plots of median test:\n' \
    'plotTest(mean_regionsFeatures_test_pairs)\n'\
    'plotSummary(mean_regionsFeatures_test_pairs,groupby="feature",align_lab="Center")\n'\
    ''
    
    with open(os.path.join(data_path, 'r_code.R'), 'w') as f:
        f.write(to_print)


def plot_aggregated_sim_data():
    import matplotlib.pyplot as plt
    plot_path = 'Data/exponential'
    path = 'Data/nonb_centered_exponential_sims'
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    # cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
    #        ['forward_'+str(i) for i in range(100)] + \
    #        ['reverse_' + str(i) for i in range(100)] + \
    #        ['mask_' + str(i) for i in range(100)]
    
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple' ]
    x = range(0, 100)
    direction = ['same', 'opposite']
    direction_cols = {'same': ['forward_'+str(i) for i in range(100)], 'opposite': ['reverse_'+str(i) for i in range(100)]}
    direction_names = {'same': 'Forward', 'opposite': 'Reverse'}
    
    for elem in elements:
        this_path = os.path.join(path, elem + '_centered.csv')
        df = pd.read_csv(this_path, index_col=0)
        for dir in direction:
            cols = direction_cols[dir]
            signal = df.loc[:, cols].to_numpy()
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            for quant_id in range(len(quantiles)):
                quant = quantiles[quant_id]
                color = colors[quant_id]
                signal_quantile = np.quantile(signal, quant, axis=0)
                # control_signal_quantile = np.quantile(control_signal, quant, axis=0)
                ax.plot(x, signal_quantile, color=color, label=str(quant), linewidth=5)
                # ax.plot(x, control_signal_quantile, color=color, linestyle='--', linewidth=4)
            # max_val = np.max(np.quantile(signal, 0.95, axis=0))
            ax.set_xlabel('Position in Window', fontsize=18)
            ax.set_ylabel('Translocation Time', fontsize=18)
            leg = plt.legend(loc=(1.03, 0.5), title="Quantile", prop={'size': 14},
                             title_fontsize=15)
            ax.add_artist(leg)
            h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
            plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
                       title="Structure", prop={'size': 14}, title_fontsize=15)
            plt.setp(ax.get_xticklabels(), Fontsize=13)
            plt.setp(ax.get_yticklabels(), Fontsize=13)
            # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
            # ax.set_ylim(top=max_val+0.00053)
            ax.set_title(elem + ' ' + direction_names[dir], fontsize=20)
            plt.tight_layout()
            plt.show()
            plt.savefig(os.path.join(plot_path, elem + direction_names[dir] + '.png'))


def reduce_win_size_center():
    # exp data
    win_size = 50
    exp_data_100_path = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/bdna'
    exp_data_50_path = '/home/mah19006/projects/nonBDNA/data/experimental_50/'
    
    if not os.path.exists(exp_data_50_path):
        os.mkdir(exp_data_50_path)
    
    np_files = [os.path.join(exp_data_100_path, name) for name in os.listdir(exp_data_100_path) if '.npy' in name]
    
    for file in np_files:
        win_np = np.load(file)
        _, file_name = os.path.split(file)
        current_size = win_np.shape[1]
        start_col_idx = (current_size - win_size) // 2
        new_np = win_np[:, start_col_idx:start_col_idx + win_size]
        np.save(os.path.join(exp_data_50_path, file_name), new_np)
    
    ###############################################################################################################
    # sim data:
    sim_data_100_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/exponential_simulated_data/'
    sim_data_50_path = '/home/mah19006/projects/nonBDNA/data/sim_50/'
    
    if not os.path.exists(sim_data_50_path):
        os.mkdir(sim_data_50_path)
    
    np_files = [os.path.join(sim_data_100_path, name) for name in os.listdir(sim_data_100_path) if '.npy' in name]
    for file in np_files:
        win_np = np.load(file)
        _, file_name = os.path.split(file)
        current_size = win_np.shape[1]
        start_col_idx = (current_size - win_size) // 2
        new_np = win_np[:, start_col_idx:start_col_idx + win_size]
        np.save(os.path.join(sim_data_50_path, file_name), new_np)
    
    
    ################################################################################################
    # experimental pandas:
    current_path_exp = '/home/mah19006/projects/nonBDNA/data/prepared_windows_req5/dataset/outliers/centered_windows'
    
    pd_files = [os.path.join(current_path_exp, name) for name in os.listdir(current_path_exp) if '.csv' in name]
    
    for file in pd_files:
        _, file_name = os.path.split(file)
        this_df = pd.read_csv(file, index_col=0)
        current_size = (len(list(this_df.columns.values)) - 7) // 3
        cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
               ['forward_'+str(i) for i in range(current_size)] + \
               ['reverse_' + str(i) for i in range(current_size)] + \
               ['mask_' + str(i) for i in range(current_size)]
        forward_cols = cols[7:current_size+7]
        reverse_cols = cols[current_size+7: 2*current_size+7]
        mask_cols = cols[2*current_size+7:]
        start_col_idx = (current_size - win_size) // 2
        forward_sel = forward_cols[start_col_idx:start_col_idx+win_size]
        reverse_sel = reverse_cols[start_col_idx:start_col_idx+win_size]
        mask_sel = mask_cols[start_col_idx:start_col_idx+win_size]
        new_cols = ['forward_'+str(i) for i in range(win_size)] + \
                   ['reverse_' + str(i) for i in range(win_size)] + \
                   ['mask_' + str(i) for i in range(win_size)]
        id_cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion']
        selected_signal_cols = forward_sel + reverse_sel + mask_sel
        selected_signal_np = this_df[selected_signal_cols].to_numpy()
        new_id_df = this_df[id_cols]
        new_signal_df = pd.DataFrame(data=selected_signal_np, columns=new_cols, index=range(selected_signal_np.shape[0]))
        new_df = pd.concat([new_id_df, new_signal_df], axis=1)
        new_df.to_csv(os.path.join(exp_data_50_path, file_name))
    ################################################################################################
    # simulation pandas:
    current_path_sim = '/labs/Aguiar/non_bdna/annotations/exponential_simulated_data/dataset'
    pd_files = [os.path.join(current_path_sim, name) for name in os.listdir(current_path_sim) if '.csv' in name]
    
    for file in pd_files:
        _, file_name = os.path.split(file)
        this_df = pd.read_csv(file, index_col=0)
        current_size = (len(list(this_df.columns.values)) - 7) // 3
        cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion'] + \
               ['forward_'+str(i) for i in range(current_size)] + \
               ['reverse_' + str(i) for i in range(current_size)] + \
               ['mask_' + str(i) for i in range(current_size)]
        forward_cols = cols[7:current_size+7]
        reverse_cols = cols[current_size+7: 2*current_size+7]
        mask_cols = cols[2*current_size+7:]
        start_col_idx = (current_size - win_size) // 2
        forward_sel = forward_cols[start_col_idx:start_col_idx+win_size]
        reverse_sel = reverse_cols[start_col_idx:start_col_idx+win_size]
        mask_sel = mask_cols[start_col_idx:start_col_idx+win_size]
        new_cols = ['forward_'+str(i) for i in range(win_size)] + \
                   ['reverse_' + str(i) for i in range(win_size)] + \
                   ['mask_' + str(i) for i in range(win_size)]
        id_cols = ['id', 'chr', 'strand', 'start', 'end', 'label', 'motif_proportion']
        selected_signal_cols = forward_sel + reverse_sel + mask_sel
        selected_signal_np = this_df[selected_signal_cols].to_numpy()
        new_id_df = this_df[id_cols]
        new_signal_df = pd.DataFrame(data=selected_signal_np, columns=new_cols, index=range(selected_signal_np.shape[0]))
        new_df = pd.concat([new_id_df, new_signal_df], axis=1)
        new_df.to_csv(os.path.join(sim_data_50_path, file_name))


def plot_experimental_one_method():
    import matplotlib.pyplot as plt
    main_path = 'Data/methods_results/IF/IF_final_results.csv'
    results_df = pd.read_csv(main_path, index_col=0)
    upper_tail_results = results_df[results_df['tail'] == 'upper'].reset_index(drop=True)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z-DNA']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    
    # ['tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
    
    plot_path = 'Data/methods_results/IF/'
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        df = upper_tail_results[upper_tail_results['label'] == elem].reset_index(drop=True)
        tuples = list(zip(df.alpha, df.potential_nonb_counts))
        fdrs = [i[0] for i in tuples]
        n_nonb = [i[1] for i in tuples]
        color = colors[elem_id]
        ax.plot(fdrs, n_nonb, color=color, linewidth=3, label=elem_name)
    ax.set_xlabel('FDR', fontsize=27)
    ax.set_ylabel('Non-B DNA Count', fontsize=27)
    leg = plt.legend(loc=(0.03, 0.55), title='Non-B Structures', prop={'size': 16},
                     title_fontsize=18)
    ax.add_artist(leg)
    # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
    # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
    #            title="Structure", prop={'size': 14}, title_fontsize=15)
    plt.setp(ax.get_xticklabels(), Fontsize=22)
    plt.setp(ax.get_yticklabels(), Fontsize=22)
    # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
    # ax.set_ylim(top=max_val+0.00053)
    # ax.set_title(elem_name + ' ' + direction_names[dir], fontsize=20)
    # plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plot_path, 'IF.png'))


def aggregate_results_exp():
    main_path = 'D:/UCONN/nonBDNA/Data/methods_results'
    exp_results_path = os.path.join(main_path, 'exp')
    # exp_results = os.listdir(exp_results_path)
    
    IF_exp_path = os.path.join(exp_results_path, 'IF_50')
    final_if_path = [os.path.join(IF_exp_path, name) for name in os.listdir(IF_exp_path) if 'final_results.csv' in name][0]
    if_results = pd.read_csv(final_if_path, index_col=0)
    if_results['method'] = 'Isolation Forest'
    ################################################################
    SVM_path = os.path.join(exp_results_path, 'SVM_50')
    svm_files = [os.path.join(SVM_path, name) for name in os.listdir(SVM_path) if 'csv' in name]
    final_svm = pd.DataFrame(columns=['method', 'label', 'alpha', 'tail', 'potential_nonb_counts'])
    for sfile in svm_files:
        df = pd.read_csv(sfile, index_col=0)
        final_svm = final_svm.append(df).reset_index(drop=True)
    final_svm['method'] = 'One Class SVM'

    ##############################################################################
    gofae_path = os.path.join(exp_results_path, 'GoFAE')
    gofae_path_file = os.path.join(gofae_path, os.listdir(gofae_path)[0])
    gofae_df = pd.read_csv(gofae_path_file, index_col=0)
    gofae_df = gofae_df.drop(['MD_rec'], axis=1)
    gofae_df['method'] = 'GoFAE-COD'
    ##################################################################################
    
    lof_path = os.path.join(exp_results_path, 'LOF_50')
    lof_files = [os.path.join(lof_path, name) for name in os.listdir(lof_path) if '_final_results.csv' in name]
    final_lof = pd.DataFrame(columns=['method', 'label', 'alpha', 'tail', 'potential_nonb_counts'])
    for lfile in lof_files:
        df = pd.read_csv(lfile, index_col=0)
        final_lof = final_lof.append(df).reset_index(drop=True)
    final_lof['method'] = 'LOF'
    ###################################################################################
    # vanilla 128
    valilabs128_path = os.path.join(exp_results_path, 'vanilla_bs128', 'saved_model_bs128')
    valilabs128 = [os.path.join(valilabs128_path,name) for name in os.listdir(valilabs128_path) if 'final_results.csv' in name][0]
    valilabs128_df = pd.read_csv(valilabs128, index_col=0)
    valilabs128_df['method'] = 'Outlier AE (128)'


    ###################################################################################
    # vanilla 64
    valilabs64_path = os.path.join(exp_results_path, 'vanilla_bs64', 'saved_model')
    valilabs64 = [os.path.join(valilabs64_path,name) for name in os.listdir(valilabs64_path) if 'final_results.csv' in name][0]
    valilabs64_df = pd.read_csv(valilabs64, index_col=0)
    valilabs64_df['method'] = 'Outlier AE (64)'
    
    #######################################################################################
    # total
    final_results = pd.DataFrame(columns=['method', 'label', 'alpha', 'tail', 'potential_nonb_counts'])
    
    final_results = final_results.append(if_results).reset_index(drop=True)
    final_results = final_results.append(final_svm).reset_index(drop=True)
    # final_results = final_results.append(final_lof).reset_index(drop=True)
    final_results = final_results.append(valilabs64_df).reset_index(drop=True)
    final_results = final_results.append(valilabs128_df).reset_index(drop=True)

    final_results = final_results.append(gofae_df).reset_index(drop=True)
    final_results = final_results.dropna().reset_index(drop=True)
    results_df = final_results
    return results_df



def plot_experimental_methods_comparison():
    import matplotlib.pyplot as plt
    results_df = aggregate_results_exp()
    plot_path = 'Data/methods_results/plots/'
    tail_types = ['upper', 'lower']
    for tail_type in tail_types:
        upper_tail_results = results_df[results_df['tail'] == tail_type].reset_index(drop=True)
        # upper_tail_results = upper_tail_results[upper_tail_results['alpha'] <= 0.5].reset_index(drop=True)
        elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                    'Short_Tandem_Repeat', 'Z_DNA_Motif']
        elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                         'Short Tandem Repeat', 'Z-DNA']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        
        # ['tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
        
        for elem_id in range(len(elements)):
            elem_name = elements_name[elem_id]
            elem = elements[elem_id]
            df = upper_tail_results[upper_tail_results['label'] == elem].reset_index(drop=True)
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            methods = list(df['method'].unique())
            methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM':'tab:orange', 'GoFAE-COD':'tab:green', 'Outlier AE (64)':'tab:red' , 'Outlier AE (128)':'tab:purple'}
            for idx, method in enumerate(methods):
                this_df = df[df['method'] == method].reset_index(drop=True)
                tuples = list(zip(this_df.alpha, this_df.potential_nonb_counts))
                fdrs = [i[0] for i in tuples]
                n_nonb = [i[1] for i in tuples]
                color = methods_colors[method]
                ax.plot(fdrs, n_nonb, color=color, linewidth=3, marker='o', label=method)
            
            ax.set_xlabel('FDR', fontsize=27)
            ax.set_ylabel('Non-B DNA Count', fontsize=27)
            # leg = plt.legend(loc=(0.9, 0.03), title='Method', prop={'size': 16}, title_fontsize=18)
            # ax.add_artist(leg)
            # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
            # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
            #            title="Structure", prop={'size': 14}, title_fontsize=15)
            plt.setp(ax.get_xticklabels(), Fontsize=22)
            plt.setp(ax.get_yticklabels(), Fontsize=22)
            # ax.text(40, max_val+0.0001, direction_names[dir], fontsize=25, bbox={'alpha': 0, 'pad': 2})
            # ax.set_ylim(top=max_val+0.00053)
            ax.set_title(elem_name + ', '+ tail_type,fontsize=20)
            # plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, 'exp_' + elem + '_' + tail_type + '.png'))



def aggregate_results_sim():
    main_path = 'D:/UCONN/nonBDNA/Data/methods_results'
    exp_results_path = os.path.join(main_path, 'sim')
    # exp_results = os.listdir(exp_results_path)
    sel_cols = ['accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr', 'data', 'label', 'method']
    
    IF_exp_path = os.path.join(exp_results_path, 'IF_sim')
    final_if_path = [os.path.join(IF_exp_path, name) for name in os.listdir(IF_exp_path) if 'final_results' in name][0]
    if_results = pd.read_csv(final_if_path, index_col=0)
    if_results['method'] = 'Isolation Forest'
    if_results = if_results[sel_cols]
    ################################################################
    SVM_path = os.path.join(exp_results_path, 'SVM_sim')
    svm_files = [os.path.join(SVM_path, name) for name in os.listdir(SVM_path) if 'csv' in name]

    measures = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'f-score', 'fpr', 'fnr', 'fdr']
    final_svm = pd.DataFrame(columns=measures + ['data', 'kernel', 'label', 'method'])
    
    
    for sfile in svm_files:
        df = pd.read_csv(sfile, index_col=0)
        final_svm = final_svm.append(df).reset_index(drop=True)
    final_svm['method'] = 'One Class SVM'
    final_svm = final_svm[sel_cols]

    ##############################################################################
    gofae_path = os.path.join(exp_results_path, 'GoFAE')
    gofae_path_file = os.path.join(gofae_path, os.listdir(gofae_path)[0])
    gofae_df = pd.read_csv(gofae_path_file, index_col=0)
    gofae_df = gofae_df.drop(['MD_rec'], axis=1)
    gofae_df['method'] = 'GoFAE-COD'
    
    ##################################################################################
    
    lof_path = os.path.join(exp_results_path, 'LOF_sim')
    lof_files = [os.path.join(lof_path, name) for name in os.listdir(lof_path) if '_final_results.csv' in name]
    final_lof = pd.DataFrame(columns=['method', 'label', 'alpha', 'tail', 'potential_nonb_counts'])
    for lfile in lof_files:
        df = pd.read_csv(lfile, index_col=0)
        final_lof = final_lof.append(df).reset_index(drop=True)
    final_lof['method'] = 'LOF'
    ###################################################################################
    # vanilla 128
    valilabs128_path = os.path.join(exp_results_path, 'vanilla_bs128', 'saved_model_bs128')
    valilabs128 = [os.path.join(valilabs128_path,name) for name in os.listdir(valilabs128_path) if 'final_results.csv' in name][0]
    valilabs128_df = pd.read_csv(valilabs128, index_col=0)
    valilabs128_df['method'] = 'Outlier AE (128)'
    
    
    ###################################################################################
    # vanilla 64
    valilabs64_path = os.path.join(exp_results_path, 'vanilla_bs64', 'saved_model')
    valilabs64 = [os.path.join(valilabs64_path,name) for name in os.listdir(valilabs64_path) if 'final_results.csv' in name][0]
    valilabs64_df = pd.read_csv(valilabs64, index_col=0)
    valilabs64_df['method'] = 'Outlier AE (64)'
    
    #######################################################################################
    # total
    final_results = pd.DataFrame(columns=sel_cols)
    
    final_results = final_results.append(if_results).reset_index(drop=True)
    final_results = final_results.append(final_svm).reset_index(drop=True)
    # final_results = final_results.append(final_lof).reset_index(drop=True)
    final_results = final_results.append(valilabs64_df).reset_index(drop=True)
    final_results = final_results.append(valilabs128_df).reset_index(drop=True)
    
    final_results = final_results.append(gofae_df).reset_index(drop=True)
    final_results = final_results.dropna().reset_index(drop=True)
    results_df = final_results
    return results_df



def plot_sim_comparison():

    df = aggregate_results_sim()
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z-DNA']
    metric = 'f-score' # 'accuracy'
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM':'tab:orange', 'GoFAE-COD':'tab:green', 'Outlier AE (64)':'tab:red' , 'Outlier AE (128)':'tab:purple'}

    df = df[df['data'] == 'test'].reset_index(drop=True)
    methods = list(df['method'].unique())
    new_df = pd.DataFrame(columns=['Non-B DNA Structure'] + methods, index=range(len(elements)))
    for i, nonb in enumerate(elements):
        
        new_df.loc[i, 'Non-B DNA Structure'] = nonb
        for meth in methods:
            if len(df.loc[((df['label'] == nonb)&(df['method'] == meth))][metric] > 0):
                new_df.loc[i, meth] = df.loc[((df['label'] == nonb)&(df['method'] == meth))][metric].values[0]
            else:
                new_df.loc[i, meth] = 0
    new_df['GoFAE-COD'] = [0.99954, 0.96700, 0.99929, 0.99901, 0.99825, 0.99906, 0.99602]
    new_df['Outlier AE (64)'] = [0.99990, 0.99990, 0.99998, 0.99999, 0.99990, 0.99989, 1]
    new_df['Outlier AE (128)'] = [0.99999, 0.9997, 0.99999, 1, 1, 0.99997, 1]
    gof = np.mean([0.99954, 0.96700, 0.99929, 0.99901, 0.99825, 0.99906, 0.99602])
    ae64 = np.mean([0.99990, 0.99990, 0.99998, 0.99999, 0.99990, 0.99989, 1])
    ae128 = np.mean([0.99999, 0.9997, 0.99999, 1, 1, 0.99997, 1])
    print('gof:', gof)
    print('ae64:', ae64)
    print('ae128:', ae128)
    
    
    new_df = new_df[['GoFAE-COD', 'Outlier AE (64)', 'Outlier AE (128)']]
    plot_path = 'Data/methods_results/plots/'
    methods = ['GoFAE-COD', 'Outlier AE (64)', 'Outlier AE (128)']
    ind = np.arange(len(new_df))
    width = 0.3

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))
    for i, method in enumerate(methods):
        color = methods_colors[method]
        ax.barh(ind + (i*width), new_df[method], 0.2, color=color, label=method)
    ax.set(yticks=ind + width, yticklabels=elements_name, ylim=[2*width - 1, len(new_df)], xlim=[0.9, 1])
    
    # ax.set(yticks=ind + width, yticklabels=new_df['Non-B DNA Structure'], ylim=[2*width - 1, len(new_df)])
    # ax.legend()
    # leg = plt.legend(bbox_to_anchor=(1.01, 0.55), title="Method", prop={'size': 14}, title_fontsize=15)
    # ax.add_artist(leg)
    ax.legend(loc='lower left', bbox_to_anchor=(1.01, 0.7), title="Method", prop={'size': 14}, title_fontsize=15)
    ax.set_xlabel('F1 Score', fontsize=27)
    # ax.set_ylabel('Translocation Time', fontsize=27)
    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=45, horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), Fontsize=22)
    plt.setp(ax.get_yticklabels(), Fontsize=22)
    plt.tight_layout()

    # plt.show()
    
    plt.savefig(os.path.join(plot_path, 'sim_' + metric + '.png'))




if __name__ == '__main__':
    if '-c' in sys.argv:
        
        chromo = sys.argv[sys.argv.index('-c') + 1]
    else:
        chromo = ''
    # make_IWT_data_extra_features_ready_paper(chromo)
    # make_IWT_data_extra_features()
    # remove_overlapping_windows()

    # make_signals_numpy(chromo)
    # aggregate_translocation_signals_numpy(chromo)
    # plot_non_b_types()
    # aggregate_translocation_signals_pandas()
    # plot_non_b_types_direction_quantiles()
    # plot_non_b_types_direction_quantiles_vs_control()
    # plot_non_b_types_direction_quantiles_vs_control_new_data()
    # make_iwt_paper_req5_chromosome_aware()
    prepare_iwt_R_script()
    