import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import os
from utils import prepare_bdna_dataset
from multiprocessing import Pool

def convert_to_pA_numpy(d, digitisation, fast5_range, offset):
    raw_unit = fast5_range / digitisation
    return (d + offset) * raw_unit

def plot_coverage(path):
    path = 'Data\chr22.txt'
    chromosome_df = pd.read_csv(path, names=['chr', 'position', 'coverage'])
    chromosome_df.hist(column=['coverage'])
    x = list(chromosome_df['coverage'])
    
    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    plt.subplot(211)
    hist, bins, _ = plt.hist(x, bins=8)
    
    logbins = np.logspace(0, np.log10(max(x)), num=100)
    name='Data\chr22_depth.png'
    plt.hist(x, bins=logbins)
    plt.xscale('log')
    plt.xlabel('Depth (log scale)')
    plt.ylabel('# reads')
    plt.tight_layout()
    plt.savefig(name)
    # plt.show()


def plot_density():
    path = 'Data\chr22.csv'
    chromosome_df = pd.read_csv(path, names=['chr', 'feature', 'start', 'end', 'strand', 'attribute'])
    chromosome_df = chromosome_df.drop(columns=['attribute'])
    chromosome_df = chromosome_df.sort_values(by='start').reset_index(drop=True)
    chromosome_df['length'] = chromosome_df.apply(lambda row: row.end - row.start, axis=1)
    # chromosome_df['strand'].unique()
    chromosome_df = chromosome_df[chromosome_df['length'] <= 150]
    p = sns.displot(chromosome_df, x="length", hue="feature", kind="kde", fill=True)
    # p.set_xlabel('Length', fontsize=20)
    # plt.show()
    # p.set(xlabel='Length',  title='Potential Non-B DNA structures on Chromosome 22\n(motifs with length >
    # 150 (0.14% of motifs) are cut out )')
    p.set(xlabel='Length')
    p._legend.set_title('Non-B DNA structures')
    plt.tight_layout()
    plt.savefig('Data\chr22_motifs_density.png')


def plot_pie_chart():
    path = 'Data\chr22.csv'
    chromosome_df = pd.read_csv(path, names=['chr', 'feature', 'start', 'end', 'strand', 'attribute'])
    chromosome_df = chromosome_df.drop(columns=['attribute'])
    chromosome_df = chromosome_df.sort_values(by='start').reset_index(drop=True)
    chromosome_df['length'] = chromosome_df.apply(lambda row: row.end - row.start, axis=1)
    ref_df = chromosome_df[chromosome_df['strand'] == '+']
    rev_df = chromosome_df[chromosome_df['strand'] == '-']
    total_count = chromosome_df.groupby(['feature'])['feature'].count()
    ref_count = ref_df.groupby(['feature'])['feature'].count()
    rev_count = rev_df.groupby(['feature'])['feature'].count()
    # values = list(total_count.values/sum(total_count.values))
    # percents = [str(round(i*100, 3))+'%' for i in list(total_count.values/sum(total_count.values))]
    # colors = ['b', 'g', 'r', 'c', 'm', 'y']
    # labels = list(total_count.index)
    plt.cla()
    plt.clf()
    plt.close()
    labels = ['Z_DNA_Motif', 'Direct_Repeat', 'G_Quadruplex_Motif',  'Mirror_Repeat',  'Short_Tandem_Repeat',
              'A_Phased_Repeat', 'Inverted_Repeat']
    values = total_count[labels].values
    explode = (0, 0, 0, 0, 0, 0, 0)
    plt.pie(values, labels=labels, explode=explode, counterclock=False, shadow=False, autopct='%1.1f%%')
    # plt.title('Non-B DNA structures type')
    # plt.legend(labels, loc="upper right", bbox_to_anchor=(0.2,0.7))# bbox_to_anchor=(1.04,1)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Data\chr22_motifs_marginal_pie.png')


def plot_pie_chart_final():
    
    def func(pct, labels_dict):
        all_motifs = np.sum([labels_dict[lab] for lab in labels])
        this_val = int(pct*all_motifs/100)
        return '{:d}'.format(this_val)
    
    plt.cla()
    plt.clf()
    plt.close()
    labels_dict = {'A Phased Repeat': 3305, 'Direct Repeat': 8542, 'G Quadruplex Motif': 4658,
                   'Short Tandem Repeat': 11548, 'Inverted Repeat': 8474, 'Mirror Repeat': 8855,
                   'Z DNA Motif': 6638, 'Control': 3538}
    labels = list(labels_dict.keys())
    values = [labels_dict[lab] for lab in labels]
    explode = (0, 0, 0, 0, 0, 0, 0, 0)
    plt.pie(values, labels=labels, explode=explode, counterclock=False, shadow=False,
            autopct=lambda pct: func(pct, labels_dict), wedgeprops={'alpha':0.7}) # , autopct='%1.1f%%'
    # plt.title('Non-B DNA structures type')
    # plt.legend(labels, loc="upper right", bbox_to_anchor=(0.2,0.7))# bbox_to_anchor=(1.04,1)
    plt.tight_layout()
    # plt.show()
    plt.savefig('Data\chr22_all_motifs_pie.png')


def plot_events(fast5_path):
    
    def convert_to_pA_numpy(d, digitisation, range, offset):
        raw_unit = range / digitisation
        return (d + offset) * raw_unit
    
    f = h5py.File(fast5_path, 'r')
    
    raw = list(f['Raw/Reads/Read_100/Signal'])
    events = list(f['Analyses/Basecall_1D_000/BaseCalled_template/Events'])
    this_raw = raw[363:]
    move_table = [ev[5] for ev in events]
    this_raw_pa = convert_to_pA_numpy(np.array(this_raw), 8192.0, 1455.31, 2)
    
    sta, end = 0, 15
    seq = 'TTGTTGTACTTCGTTCGGTTACGTATTGCTTTGATTCCATTTTGATGATTATTCATTCTATTCATTCGGTGGTTCATTCAATTCCATTTGTTGATGATTCCATTGATGATTCCATTCCATTCCATTCGATGATGATTCAATCGAGTCCGATGATTCCTTCGATTCCATTCGATGATGATTCCATTAAAGTCCATTCAATGATTCCATTCGATTCCGATTCAATGATGATTCCTTTGATCCATTCAATGATACCATTTGATTCCATTCGATGATGATTCCATTTGAGTGCACTCATAATTCCATTATTCATTTCCGTTCGATGATTGTTCATTCAATGCCATTCCTTGATGAATCCTTTCCATTCATTGATGATGACTCCTTTTGATTACATTTGTTGATCTTCCATTCGATTCCATTAGTGATGACTCCATTCAATGATGATTCCATTCAATTCGATAGATGATTCCTTTCAATGATTATTCCATTCAATTCCATTCCATGATTATTCCCTTCGATTCATTTGATGATGGTTCCATTCGAATTCCATTAGATGATAGTCCACTCAAGTCCGTTTAATGATTCCATTAAACTTGCATTCAATCATTCCATTGAGTCCATTCAAAGATTCGTTAGTTTCCATTAGATGATGATACCAATTCGAGTCCATTCCATGATTCCATTCAATTCCATTTGATGATGATTCCATTCGAGTCCTCAATGATTCCATTGATTCTATAGATGATGATTCCATTCGATTCCATTCCTACGATGATCATTCCATTGATGCCATTCAATGATTCCATTCATGTCCATTCGATGATTATTCCATTTGAGTTCATTCGATGAGTCCATTCGATTCCATTAGATGATGATTCCATTCGTTCCATTCAATGGTGACTCATTCGTGCATTTGATGATGACTCCTTCAATTCATTCTATGATTCCATTTAATTCCGTTCGATGATGATTCCATTCGATCCATTCGATGATTTCATTCGATTCATTCTATTATTCAATTCGTTTCCATTCGATGATGATTCCATTCGATTCCATTCAATGATAATTAGGTCGATTACTTTTGATGATTCTATTCGATTCCATTGATGATTCCTTCAAACCCGTTTGATGATTTCATTCGAGTCCATTCGATGACTCCATACGATTCCGTTCGATGATTCCATTGGAATGGTGATGATGATTCCATTCGTTCCATTTGATGATTCCATTCGATTCCATTCGATGATTCCATGATTCCATTGATGATGATTCCATTCGATTCCATTGATTCCATTCAATCCATTTGATGGCGATTCCATTCGATTCCATTCGATGATGATTCCATTAGTTTCCATTCGATGATTCATTCAATTCCATTAATTGATGATTATTCTATTGATTCCATTCGATGATGATTCCATTGATTCCATTCGATATTCCTTCGATTCCATTTGATGATTCACTTGATTCCTTTTGATGATGATTCCATTCGATTCCATTCATTGATTCAATGATTCTATTCGATGATCATTCCATTCGGCTCCATTCAATGATGATTCCATTCGATCCATTCGAATTCCATTTGATTCCATTTGATATTGATTCCATTGATACCATTCTGTGATCCCTCTATTCCATTTAATGATGATTCCATTCGATTCCATTTGATAATTCCATTCGATTCCATATGATGATTATTCCATTCGTTTCCGTTTGATGATTCCATTCGATTTCATTCAATGATGTTTCCATTAGAGTCCATTCAATGATTCCATTCCATTCCATTCGATGATTATTCCATTAGAGTTCATTAGAAGATTCCTTTCAATTCAATTTGTTGATGATTCCATTCGAGGGA'
    fig, ax = plt.subplots(figsize=(15, 3))
    seq_id = sta-1/5
    for i in np.arange(sta, end):
        first_i = i*5
        second_i = (i+1)*5
        
        if move_table[i] == 1:
            seq_id += 1/5
            this_seq = seq[int(seq_id*5):int((seq_id+1)*5)]
            
            b = r""+this_seq[0:4]+"" + r"$\bf{"+this_seq[-1]+"}$"
            
            ax.axvspan(i*5, (i+1)*5-1, facecolor='gray', alpha=0.2)
            plt.text(i*5, 70, b, size=12, rotation=0,
                     ha="left", va="center",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)))
        
        if move_table[i] == 2:
            seq_id += 2/5
            this_seq = seq[int(seq_id*5):int((seq_id+1)*5)]
            b = r""+this_seq[0:3]+"" + r"$\bf{"+this_seq[-2:]+"}$"
            ax.axvspan(i*5, (i+1)*5-1, facecolor='gray', alpha=0.5)
            plt.text(i*5, 70, b, size=12, rotation=0,
                     ha="left", va="center",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)))
        
        
        scatter0 = np.arange(first_i, second_i), this_raw_pa[first_i:second_i]
        line0 = np.arange(first_i, second_i), [np.mean(this_raw_pa[first_i:second_i])]*5
        # line1 = np.arange(first_i, second_i), [events[i][0]] * 5
        # move_table[i]
        ax.plot(line0[0], line0[1], 'r')
        # plt.plot(line1[0], line1[1], 'b')
        ax.scatter(scatter0[0], scatter0[1], c='black', s=10)
        ax.set_xlabel('time (s) x Sampling Rate (4000)')
        ax.set_ylabel('Current (pA)')
    plt.tight_layout()
    # plt.show()
    fig.savefig('event_moves.png')


def plot_correlation_histograms():
    # std_df_merged = pd.read_csv('all_std_df.csv')
    corr_df_merged_current = pd.read_csv('summaries\current_merged_corr_df.csv')
    corr_df_merged_duration = pd.read_csv('summaries\duration_merge_corr_df.csv')
    
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()
    plt.rcParams["legend.markerscale"] = 3
    f, axes = plt.subplots(1, 2, figsize=(18, 7))
    g = sns.histplot(data=corr_df_merged_current, x="Value", hue="Structure", kde=True, ax=axes[0])
    # legends = axes[0].get_legend()
    # legends.set_bbox_to_anchor([0.92, 0.80])
    plt.setp(axes[0].get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(axes[0].get_legend().get_title(), fontsize='20')
    g.set_yticklabels([int(i) for i in g.get_yticks()], size=15)
    g.set_xticklabels(['{:.1f}'.format(i) for i in g.get_xticks()], size=15)
    axes[0].set_xlabel(r'Current Correlations ($\rho_{\mathcal{C}}$)', fontsize=20)
    axes[0].set_ylabel('Count', fontsize=20)
    
    g2 = sns.histplot(data=corr_df_merged_duration, x="Value", hue="Structure", kde=True, ax=axes[1])
    # legends = axes[1].get_legend()
    # legends.set_bbox_to_anchor([0.92, 0.80])
    plt.setp(axes[1].get_legend().get_texts(), fontsize='20') # for legend text
    plt.setp(axes[1].get_legend().get_title(), fontsize='20')
    g2.set_yticklabels([int(i) for i in g2.get_yticks()], size=15)
    g2.set_xticklabels(['{:.1f}'.format(i) for i in g2.get_xticks()], size=15)
    axes[1].set_xlabel(r'Duration Correlations ($\rho_{\Delta}$)', fontsize=20)
    axes[1].set_ylabel('Count', fontsize=20)
    plt.tight_layout()
    # plt.show()
    f.savefig('correlation_hist.png')


def plot_std():
    std_df_merged_current = pd.read_csv('summaries\current_merged_std_df.csv')
    std_df_merged_duration = pd.read_csv('summaries\duration_merge_std_df.csv')
    plt.close('all')
    plt.close()
    plt.clf()
    plt.cla()
    f, axes = plt.subplots(1, 2, figsize=(18, 7))
    g = sns.lineplot(x="Position", y="Value", hue="Structure",  data=std_df_merged_current, ax=axes[0])
    legends = g.get_legend()
    legends.set_bbox_to_anchor([1.0, 1.0])
    plt.setp(axes[0].get_legend().get_texts(), fontsize='20')
    plt.setp(axes[0].get_legend().get_title(), fontsize='20')
    g.set_yticklabels(['{:.2f}'.format(i) for i in g.get_yticks()], size=15)
    g.set_xticklabels([int(i) for i in g.get_xticks()], size=15)
    axes[0].set_xlabel('Position', fontsize=20)
    axes[0].set_ylabel(r'$Std_{\mathcal{C}}$', fontsize=20)
    
    g = sns.lineplot(x="Position", y="Value", hue="Structure",  data=std_df_merged_duration, ax=axes[1])
    g.set_yticklabels(['{:.4f}'.format(i) for i in g.get_yticks()], size=15)
    g.set_xticklabels([int(i) for i in g.get_xticks()], size=15)
    axes[1].set_xlabel('Position', fontsize=20)
    axes[1].set_ylabel(r'$Std_{\Delta}$', fontsize=20)
    axes[1].get_legend().remove()
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('std_line_plots.png')




def plot_any_event(fast5_path, fastq_path):
    fastq_file_lists = ['fastq_runid_dee6dbdbaa89158bd70425255d3b2faff3d1fbf6_0.fastq']
    # fast5_path ='fff13cfd-c555-43db-8a3c-35f93c60db3b.fast5'# no correct
    fast5_path = 'ffb4be6f-8aa9-43b5-987c-f7c87d8fad9f.fast5'

    
    f = h5py.File(fast5_path, 'r')
    read_no = list(f['Raw/Reads/'])[0]
    raw = list(f['Raw/Reads/'+read_no+'/Signal'])
    events = list(f['Analyses/Basecall_1D_001/BaseCalled_template/Events'])
    seq = f['Analyses/Basecall_1D_001/BaseCalled_template/Fastq'][()].decode().split('\n')[1]
    offset_no = events[0][1]
    this_raw = raw[offset_no:]
    move_table = [ev[5] for ev in events]
    digitisation = f['UniqueGlobalKey/channel_id'].attrs.get('digitisation')
    offset = f['UniqueGlobalKey/channel_id'].attrs.get('offset')
    fast5_range = f['UniqueGlobalKey/channel_id'].attrs.get('range')
    this_raw_pa = convert_to_pA_numpy(np.array(this_raw), digitisation, fast5_range, offset)
    
    sta, end = 0, 20
    fig, ax = plt.subplots(figsize=(15, 3))
    seq_id = sta-1/5
    for i in np.arange(sta, end):
        first_i = i*5
        second_i = (i+1)*5
        
        if move_table[i] == 1:
            seq_id += 1/5
            # this_seq = seq[int(seq_id*5):int((seq_id+1)*5)]
            this_seq = seq[int(seq_id*5):int((seq_id+1)*5)]
            
            b = r""+this_seq[0:4]+"" + r"$\bf{"+this_seq[-1]+"}$"
            
            ax.axvspan(i*5, (i+1)*5-1, facecolor='gray', alpha=0.2)
            plt.text(i*5, 60, b, size=12, rotation=0,
                     ha="left", va="center",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)))
        
        if move_table[i] == 2:
            seq_id += 2/5
            # this_seq = seq[int(seq_id*5):int((seq_id+1)*5)]
            this_seq = seq[int(seq_id*5):int((seq_id+1)*5)]
            b = r""+this_seq[0:3]+"" + r"$\bf{"+this_seq[-2:]+"}$"
            ax.axvspan(i*5, (i+1)*5-1, facecolor='gray', alpha=0.5)
            plt.text(i*5, 60, b, size=12, rotation=0,
                     ha="left", va="center",
                     bbox=dict(boxstyle="square",
                               ec=(1., 0.5, 0.5),
                               fc=(1., 0.8, 0.8)))
        
        scatter0 = np.arange(first_i, second_i), this_raw_pa[first_i:second_i]
        line0 = np.arange(first_i, second_i), [np.mean(this_raw_pa[first_i:second_i])]*5
        # line1 = np.arange(first_i, second_i), [events[i][0]] * 5
        # move_table[i]
        ax.plot(line0[0], line0[1], 'r')
        # plt.plot(line1[0], line1[1], 'b')
        ax.scatter(scatter0[0], scatter0[1], c='black', s=10)
        ax.set_xlabel('time (s) x Sampling Rate (4000)')
        ax.set_ylabel('Current (pA)')
    plt.tight_layout()
    plt.show()
    # fig.savefig('event_moves.png')
    sampling_rate = f['UniqueGlobalKey/channel_id'].attrs.get('sampling_rate')

    corrected_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
    corrected_moves = [ev[3] for ev in corrected_events]
    corrected_seq = ''.join([ev[4].decode() for ev in corrected_events])

    trans_sec = corrected_moves/sampling_rate


    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_deletions')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_insertions')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_matches')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_mismatches')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_start')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_end')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('clipped_bases_start')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('clipped_bases_end')
    f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_chrom')
    f['Analyses/RawGenomeCorrected_000'].attrs.get('basecall_group')
    f['Analyses/RawGenomeCorrected_000'].attrs.get('tombo_version')
    scale = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('scale')
    shift = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('shift')
    re_idx = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs.get('read_start_rel_to_raw')
    
    
    norm_raw = (raw - shift)/scale
    fig, ax = plt.subplots(figsize=(15, 3))

    for i in range(10, 30):

        seg_x_axis = np.arange(corrected_events[i][2], corrected_events[i][2]+corrected_events[i][3])
        seg_mean = np.mean(norm_raw[re_idx+corrected_events[i][2]:re_idx+corrected_events[i][2]+corrected_events[i][3]])
        seg_raw = norm_raw[re_idx+corrected_events[i][2]:re_idx+corrected_events[i][2]+corrected_events[i][3]]
        seg_seq = corrected_events[i][4].decode()
        seg_time = trans_sec[i]
        
        ax.plot(seg_x_axis, [seg_mean]*len(seg_x_axis), 'r')
        ax.scatter(seg_x_axis, seg_raw, c='black', s=10)
        
        plt.text((seg_x_axis[0]+seg_x_axis[-1])/2, seg_mean + 1, seg_seq, size=12, rotation=0,
                 ha="left", va="center",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8)))
    fig.savefig('tombo.png')
    plt.show()

def plot_tombo_segments_and_time():
    fast5_path = '0009a349-52eb-4500-a061-bec2b27858f7.fast5' # forward
    f = h5py.File(fast5_path, 'r')
    read_no = list(f['Raw/Reads/'])[0]
    raw = list(f['Raw/Reads/'+read_no+'/Signal'])
    events = list(f['Analyses/Basecall_1D_001/BaseCalled_template/Events'])
    offset_no = events[0][1]
    this_raw = raw[offset_no:]
    move_table = [ev[5] for ev in events]
    digitisation = f['UniqueGlobalKey/channel_id'].attrs.get('digitisation')
    offset = f['UniqueGlobalKey/channel_id'].attrs.get('offset')
    fast5_range = f['UniqueGlobalKey/channel_id'].attrs.get('range')
    this_raw_pa = convert_to_pA_numpy(np.array(this_raw), digitisation, fast5_range, offset)
    sampling_rate = f['UniqueGlobalKey/channel_id'].attrs.get('sampling_rate')

    corrected_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
    corrected_moves = [ev[3] for ev in corrected_events]
    corrected_seq = ''.join([ev[4].decode() for ev in corrected_events])
    
    trans_sec = corrected_moves/sampling_rate
    
    scale = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('scale')
    shift = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('shift')
    re_idx = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs.get('read_start_rel_to_raw')
    
    norm_raw = (raw - shift)/scale
    raw_pa = convert_to_pA_numpy(raw, digitisation, fast5_range, offset)
    
    seg_time = []
    gen_positions = []
    sequences = []
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(15, 6))

    for i in range(10, 30):
    
        seg_x_axis = np.arange(corrected_events[i][2]+re_idx, corrected_events[i][2]+corrected_events[i][3]+re_idx)
        # seg_mean = np.mean(norm_raw[re_idx+corrected_events[i][2]:re_idx+corrected_events[i][2]+corrected_events[i][3]])
        seg_mean = corrected_events[i][0]
        seg_raw = norm_raw[re_idx+corrected_events[i][2]:re_idx+corrected_events[i][2]+corrected_events[i][3]]
        seg_seq = corrected_events[i][4].decode()
        sequences.append(seg_seq)
        seg_time.append(trans_sec[i])
        gen_positions += list(seg_x_axis)
        ax1.plot(seg_x_axis, [seg_mean]*len(seg_x_axis), 'r')
        ax1.scatter(seg_x_axis, seg_raw, c='black', s=10)
    
        ax1.text((seg_x_axis[0]+seg_x_axis[-1])/2, seg_mean + 1, seg_seq, size=12, rotation=0,
                 ha="left", va="center",
                 bbox=dict(boxstyle="square",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8)))
    gaps = int(np.floor(len(gen_positions)/len(seg_time)))
    
    ax2.plot(gen_positions[::gaps][:len(seg_time)], seg_time, color='black', marker='o', markersize=10,
             markerfacecolor='pink', markeredgecolor='red')
    # labels = [item.get_text() for item in ax2.get_xticklabels()]
    # labels = sequences
    ax2.set_xticks(gen_positions[::gaps][:len(seg_time)])
    ax2.set_xticklabels(sequences)

    ax2.set_xlabel('Bases in genomic position order')
    ax2.set_ylabel('Time (sec.)')
    ax1.set_xlabel('Measurements')
    ax1.set_ylabel('Normalized Raw Signal')
    # plt.tight_layout()
    fig.savefig('tombo_time.png')
    fig.savefig('tombo_time.pdf')
    plt.show()
    

def sh_commands_print():
    
    for j in range(16):
        batch_info = '#!/bin/bash\n' \
                     '#BATCH --job-name=tomb\n' \
                     '#SBATCH -N 1\n' \
                     '#SBATCH -n 30\n' \
                     '#SBATCH -c 1\n' \
                     '#SBATCH --partition=general\n' \
                     '#SBATCH --qos=general\n' \
                     '#SBATCH --mail-type=END\n' \
                     '#SBATCH --mem=50G\n' \
                     '#SBATCH --mail-user=marjan.hosseini@uconn.edu\n' \
                     '#SBATCH -o x{}.out\n' \
                     '#SBATCH -e x{}.err\n' \
                     'echo `hostname`\n' \
                     'module load tombo;\n'.format(str(j), str(j))
        with open('xtom'+str(j).zfill(2)+'.sh', 'w', newline='\n') as file:
            file.write(batch_info)
        # print(batch_info)
            for i in range(j*150, (j+1)*150):
                comm = "tombo resquiggle '/labs/Aguiar/non_bdna/human/na12878/fast5/dist_albacore_single/{}/workspace/pass/'" \
                       " '/labs/Aguiar/non_bdna/genome_reference/GCA_000001405.29_GRCh38.p14_genomic.fna' --dna --overwrite " \
                       "--basecall-group Basecall_1D_001 --include-event-stdev " \
                       "--failed-reads-filename '/labs/Aguiar/non_bdna/human/na12878/tombo_failed_reads/{}.txt' " \
                       "--processes 30 --ignore-read-locks;\n".format(i, i)    # print(comm)
                comm2 = 'echo "tombo folder {} is done.";\n'.format(i)
                file.write(comm)
                file.write(comm2)
    
    
    
    
    batch_info = '#!/bin/bash\n' \
                 '#BATCH --job-name=tomb\n' \
                 '#SBATCH -N 1\n' \
                 '#SBATCH -n 1\n' \
                 '#SBATCH -c 1\n' \
                 '#SBATCH --partition=general\n' \
                 '#SBATCH --qos=general\n' \
                 '#SBATCH --mail-type=END\n' \
                 '#SBATCH --mem=20G\n' \
                 '#SBATCH --mail-user=marjan.hosseini@uconn.edu\n' \
                 '#SBATCH -o move.out\n' \
                 '#SBATCH -e move.err\n' \
                 'echo `hostname`\n'
    with open('move.sh', 'w', newline='\n') as file:
        file.write(batch_info)
    
        for i in range(5, 1300):
            comm = 'mv /labs/Aguiar/non_bdna/human/na12878/fast5/single/{} /home/FCAM/mhosseini/nonBDNA/data/;\n'.format(i)
            file.write(comm)
    
    
    
    path ='/labs/Aguiar/non_bdna/human/na12878/fast5/dist_albacore_single'
    missing = sorted([i for i in range(2272) if not str(i) in os.listdir(path)])
    print(missing)
    len(missing)
    
    
    for i in range(1774, 2045):
        comm = 'chmod -R 777 /labs/Aguiar/non_bdna/human/na12878/fast5/dist_albacore_single/{} &'.format(i)
        print(comm)
    
    
    batch_info = '#!/bin/bash\n' \
                 '#BATCH --job-name=tomb\n' \
                 '#SBATCH -N 1\n' \
                 '#SBATCH -n 1\n' \
                 '#SBATCH -c 1\n' \
                 '#SBATCH --partition=general\n' \
                 '#SBATCH --qos=general\n' \
                 '#SBATCH --mail-type=END\n' \
                 '#SBATCH --mem=20G\n' \
                 '#SBATCH --mail-user=marjan.hosseini@uconn.edu\n' \
                 '#SBATCH -o move.out\n' \
                 '#SBATCH -e move.err\n' \
                 'echo `hostname`\n'
    with open('move3.sh', 'w', newline='\n') as file:
        file.write(batch_info)
        
        for i in range(0, 500):
            comm = 'mv /archive/projects/nonbdna/single/{} /scratch/single/;\n'.format(i)
            file.write(comm)
    
    for i in range(0, 1000):
        comm = 'rm -r /labs/Aguiar/non_bdna/human/na12878/fast5/dist_albacore_single/{}/workspace/fail/;'.format(i)
        print(comm)
        
    path = '/labs/Aguiar/non_bdna/human/na12878/fast5/dist_albacore_single/'
    missing_workspace = []
    missing_pass = []
    missing_0 = []
    missing_fastq = []
    for i in range(0, 2272):
        print(i)
        if 'workspace' not in os.listdir(os.path.join(path, str(i))):
            missing_workspace.append(i)
        else:
            if 'pass' not in os.listdir(os.path.join(path, str(i), 'workspace')):
                missing_pass.append(i)
            else:
                if '0' not in os.listdir(os.path.join(path, str(i), 'workspace', 'pass')):
                    missing_0.append(i)
                if len([fname for fname in os.listdir(os.path.join(path, str(i), 'workspace', 'pass')) if '.fastq' in fname]) == 0:
                    missing_fastq.append(i)
    
    
    for j in [16]:
        batch_info = '#!/bin/bash\n' \
                     '#BATCH --job-name=tomb\n' \
                     '#SBATCH -N 1\n' \
                     '#SBATCH -n 30\n' \
                     '#SBATCH -c 1\n' \
                     '#SBATCH --partition=general\n' \
                     '#SBATCH --qos=general\n' \
                     '#SBATCH --mail-type=END\n' \
                     '#SBATCH --mem=50G\n' \
                     '#SBATCH --mail-user=marjan.hosseini@uconn.edu\n' \
                     '#SBATCH -o x{}.out\n' \
                     '#SBATCH -e x{}.err\n' \
                     'echo `hostname`\n' \
                     'module load tombo;\n'.format(str(j), str(j))
        with open('xtom'+str(j).zfill(2)+'.sh', 'w', newline='\n') as file:
            file.write(batch_info)
            # print(batch_info)
            for i in [370, 1206, 2071, 957, 1786, 1944, 709, 213, 1912, 1499, 1324, 2018, 688, 1743, 295]:
                comm = "tombo resquiggle '/labs/Aguiar/non_bdna/human/na12878/dist_albacore_single/{}/workspace/pass/'" \
                       " '/labs/Aguiar/non_bdna/genome_reference/GCA_000001405.29_GRCh38.p14_genomic.fna' --dna --overwrite " \
                       "--basecall-group Basecall_1D_001 --include-event-stdev " \
                       "--failed-reads-filename '/labs/Aguiar/non_bdna/human/na12878/tombo_failed_reads/{}-2.txt' " \
                       "--processes 30 --ignore-read-locks;\n".format(i, i)    # print(comm)
                comm2 = 'echo "tombo folder {} is done.";\n'.format(i)
                file.write(comm)
                file.write(comm2)

def samtools_depth_commands():
    source_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr/'
    save_path = '/labs/Aguiar/non_bdna/human/na12878/depth/'
    files = sorted([name for name in os.listdir(source_path) if '.bam' in name])
    for f in files:
        name = f.split('.bam')[0]
        comm = 'samtools depth -Q 20 {}{}.bam > {}{}.txt'.format(source_path, name, save_path, name)
        print(comm)


def plot_motif_length():
    path = 'all_non_b_windows.csv'
    chromosome_df = pd.read_csv(path)
    chromosome_df = chromosome_df.sort_values(by='start').reset_index(drop=True)
    chromosome_df = chromosome_df[chromosome_df['motif_len']<= 150].reset_index(drop=True)
    p = sns.displot(chromosome_df, x="motif_len", hue="feature", kind="kde", fill=True)
    # p.set(xlabel='Length')
    p.set(xlabel='Length', title='Potential Non-B DNA structures\n(motifs with length > 150 (0.12% of motifs) are cut out )')
    p._legend.set_title('Non-B DNA structures')
    plt.tight_layout()
    plt.savefig('overal_motif_length_density.png')


    p = sns.displot(chromosome_df, x="motif_len", hue="feature", kind="kde", col="chr", fill=True, col_wrap=5)
    # p.set_xlabel('Length', fontsize=20)
    # plt.show()
    # p.set(xlabel='Length', title='Potential Non-B DNA structures\n(motifs with length > 150 (0.12% of motifs)
    # are cut out )')
    p.set(xlabel='Length')
    p._legend.set_title('Non-B DNA structures')
    plt.tight_layout()
    plt.savefig('motif_length_density_by_chromosome.png')


def plot_MAPQ_histograms_per_chr():
    bed_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr'
    plot_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/plots/'
    all_chr = [name for name in os.listdir(bed_path) if '.bed' in name]
    for filename in all_chr:
        plot_name = filename.split('.bed')[0]
        filtered_df = pd.read_csv(os.path.join(bed_path, filename), names=['chr', 'start', 'end', 'read_id', 'score', 'strand'],
                                  index_col=False, sep='\t')
        n_duplicates = np.sum(filtered_df.duplicated())
        ratio = n_duplicates/len(filtered_df)
        unique_df = filtered_df.drop_duplicates().reset_index(drop=True)
        scores = list(unique_df['score'])
        plt.cla()
        plt.clf()
        plt.close()
        plt.hist(scores, 20)
        plt.title(plot_name + ' - number of duplicates:'+ str(n_duplicates)+ ' - ratio:'+ str(round(ratio, 4)))
        plt.savefig(os.path.join(plot_path, plot_name + '.png'))
        print(os.path.join(plot_path, plot_name + '.png'))

def plt_depth(inp):
    filename, path, plot_path = inp
    # filename = 'chr7+.txt'
    plot_name = filename.split('.txt')[0]
    df = pd.read_csv(os.path.join(path, filename), names=['chr', 'position', 'depth'], index_col=False, sep='\t')
    plt.cla()
    plt.clf()
    plt.close()
    # plt.hist(depth, 80)
    ax = df.hist(column='depth', bins=100, grid=False)
    plt.xlabel('Depth (Number of reads per position)')
    plt.ylabel('Positions Count')
    # ax.set_xlabel('Depth (Number of reads per position)')
    # ax.set_ylabel('Positions Count')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, plot_name + '.png'))
    # plt.xlabel('Depth (Number of reads per position)')
    # plt.ylabel('Positions Count')
    # plt.tight_layout()
    print(os.path.join(plot_path, plot_name + '.png'))

    
def plot_depth_histograms_per_chr():
    path = '/labs/Aguiar/non_bdna/human/na12878/depth/'
    plot_path = '/labs/Aguiar/non_bdna/human/na12878/depth/plots/'
    # all_chr = [name for name in os.listdir(path) if '.txt' in name]
    existing_plots = [name for name in os.listdir(plot_path) if '.png' in name]
    all_chr = [[name, path, plot_path] for name in os.listdir(path) if '.txt' in name and not name.split('.txt')[0]+'.png' in existing_plots]
    # print(len(all_chr))
    # pool = Pool(3)
    # pool.map(plt_depth, all_chr)
    for allc in all_chr:
        plt_depth(allc)


def plot_preprocessing_fig1():
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    plt.clf()
    
    control1_x = np.arange(51)
    motif_x = np.arange(50, 71)
    control2_x = np.arange(70, 120)

    control1_y_read1 = np.random.normal(0.02, 0.0001, len(control1_x))
    control1_y_read2 = np.random.normal(0.02, 0.0001, len(control1_x))
    control1_y_read3 = np.random.normal(0.02, 0.0001, len(control1_x))

    motif_y_read1 = norm.pdf(motif_x, 60, 15)
    motif_y_read2 = norm.pdf(motif_x, 60, 16)
    motif_y_read3 = norm.pdf(motif_x, 60, 17)
    
    motif_y_read4 = np.random.normal(0.022, 0.0002, len(motif_x))
    motif_y_read5 = np.random.normal(0.022, 0.0002, len(motif_x))
    motif_y_read6 = np.random.normal(0.022, 0.0002, len(motif_x))

    control2_y_read1 = np.random.normal(0.02, 0.0001, len(control2_x))
    control2_y_read2 = np.random.normal(0.02, 0.0001, len(control2_x))
    control2_y_read3 = np.random.normal(0.02, 0.0001, len(control2_x))


    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(control1_x, control1_y_read1, color='tab:gray')
    ax.plot(control1_x, control1_y_read2, color='tab:gray')
    ax.plot(control1_x, control1_y_read3, color='tab:gray')
    # ax.plot(motif_x, motif_y_read1, color='tab:red')
    # ax.plot(motif_x, motif_y_read2, color='tab:red')
    # ax.plot(motif_x, motif_y_read3, color='tab:red')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read1[-1]] + list(motif_y_read1) + [control2_y_read1[0]], color='w')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read2[-1]] + list(motif_y_read2) + [control2_y_read2[0]], color='w')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read3[-1]] + list(motif_y_read3) + [control2_y_read3[0]], color='w')

    ax.plot([50] + list(motif_x) + [70], [control1_y_read1[-1]] + list(motif_y_read4) + [control2_y_read1[0]], color='tab:red')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read2[-1]] + list(motif_y_read5) + [control2_y_read2[0]], color='tab:red')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read3[-1]] + list(motif_y_read6) + [control2_y_read3[0]], color='tab:red')
    
    ax.plot(control2_x, control2_y_read1, color='tab:gray')
    ax.plot(control2_x, control2_y_read2, color='tab:gray')
    ax.plot(control2_x, control2_y_read3, color='tab:gray')
    plt.ylim(0.01, 0.03)
    plt.xlim(20, 100)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        right=False,      # ticks along the bottom edge are off
        left=False,         # ticks along the top edge are off
        labelleft=False) # labels along the bottom edge are off
    # plt.show()

    plt.savefig('Figures/fig1.png')

    motif_sum = (motif_y_read4 + motif_y_read5 + motif_y_read6)/3
    cont1_sum = (control1_y_read2 + control1_y_read2 + control1_y_read3) /3
    cont2_sum = (control2_y_read2 + control2_y_read2 + control2_y_read3) /3

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(control1_x, control1_y_read1, color='w')
    ax.plot(control1_x, control1_y_read2, color='w')
    ax.plot(control1_x, control1_y_read3, color='w')

    ax.plot([50] + list(motif_x) + [70], [control1_y_read1[-1]] + list(motif_y_read1) + [control2_y_read1[0]], color='w')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read2[-1]] + list(motif_y_read2) + [control2_y_read2[0]], color='w')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read3[-1]] + list(motif_y_read3) + [control2_y_read3[0]], color='w')

    ax.plot([50] + list(motif_x) + [70], [control1_y_read1[-1]] + list(motif_y_read4) + [control2_y_read1[0]], color='w')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read2[-1]] + list(motif_y_read5) + [control2_y_read2[0]], color='w')
    ax.plot([50] + list(motif_x) + [70], [control1_y_read3[-1]] + list(motif_y_read6) + [control2_y_read3[0]], color='w')

    ax.plot(control2_x, control2_y_read1, color='w')
    ax.plot(control2_x, control2_y_read2, color='w')
    ax.plot(control2_x, control2_y_read3, color='w')


    ax.plot(control1_x, cont1_sum, color='tab:gray')
    ax.plot([50] + list(motif_x) + [70], [cont1_sum[-1]] + list(motif_sum) + [cont2_sum[0]], color='tab:red')
    ax.plot(control2_x, cont2_sum, color='tab:gray')
    
    
    plt.ylim(0.01, 0.03)
    plt.xlim(20, 100)
    plt.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        right=False,        # ticks along the bottom edge are off
        left=False,         # ticks along the top edge are off
        labelleft=False)    # labels along the bottom edge are off
    # plt.show()
    plt.savefig('Figures/fig1_input.png')

    x_all = np.arange(-10, 10, 0.001) # entire range of x, both in and out of spec
    # mean = 0, stddev = 1, since Z-transform was calculated
    y1 = norm.pdf(x_all, 0, 1)
    y2 = norm.pdf(x_all, 4, 2)
    # motif_y_read1 = norm.pdf(motif_x, 0, 0.01)
    # motif_y_read2 = norm.pdf(motif_x, 1, 0.1)
    # motif_y_read3 = norm.pdf(motif_x, 60, 17)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot([2, 2], [0, np.max(y1)], color='g', linestyle='--', linewidth=4)
    ax.plot(x_all, y1, color='tab:gray', linewidth=8)
    ax.plot(x_all, y2, color='tab:red', linewidth=8)
    plt.ylim(-0.1, 0.5)
    plt.xlim(-5, 10)
    plt.tick_params(
        axis='x',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        bottom=False,       # ticks along the bottom edge are off
        top=False,          # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.tick_params(
        axis='y',           # changes apply to the x-axis
        which='both',       # both major and minor ticks are affected
        right=False,        # ticks along the bottom edge are off
        left=False,         # ticks along the top edge are off
        labelleft=False)    # labels along the bottom edge are off
    # plt.show()
    plt.savefig('Figures/fig1_dist.png')


def plot_stats_bdna():
    forward_tr, forward_val, forward_te, reverse_tr, reverse_val, reverse_te = prepare_bdna_dataset()

    def get_stat_scaler(x):
        med = np.median(x)
        q1 = np.percentile(x, 25, interpolation='midpoint')
        q3 = np.percentile(x, 75, interpolation='midpoint')
        iqr = q3 - q1
        return med, iqr
        
    def get_stat(x):
        med = np.median(x, axis=0)
        q1 = np.percentile(x, 25, interpolation='midpoint', axis=0)
        q3 = np.percentile(x, 75, interpolation='midpoint', axis=0)
        iqr = q3 - q1
        return med, iqr
      
    medftr, iqrftr = get_stat(forward_tr)
    medfval, iqrfval = get_stat(forward_val)
    medfte, iqrfte = get_stat(forward_te)
    medrtr, iqrrtr = get_stat(reverse_tr)
    medrval, iqrrval = get_stat(reverse_val)
    medrte, iqrrte = get_stat(reverse_te)
    
    medftrscaler, iqrftrscaler = get_stat_scaler(forward_tr)
    medfvalscaler, iqrfvalscaler = get_stat_scaler(forward_val)
    medftescaler, iqrftescaler = get_stat_scaler(forward_te)
    medrtrscaler, iqrrtrscaler = get_stat_scaler(reverse_tr)
    medrvalscaler, iqrrvalscaler = get_stat_scaler(reverse_val)
    medrtescaler, iqrrtescaler = get_stat_scaler(reverse_te)
    
    positions = np.arange(100)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(positions, medftr, linewidth=1, label='Median Forward Training')
    ax.plot(positions, medfval, linewidth=1, label='Median Forward Validation')
    ax.plot(positions, medfte, linewidth=1, label='Median Forward Test')
    # ax.plot(positions, medrtr, linewidth=1, label='median reverse training')
    # ax.plot(positions, medrval, linewidth=1, label='median reverse validation')
    # ax.plot(positions, medrte, linewidth=1, label='median reverse test')
    plt.xlabel('Positions', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Value', fontsize=15)  # Add a y-label to the axes.
    ax.legend()
    title = 'Median in all positions' \
            '\nForward Median TR: {}, Forward Median val: {}, Forward Median TE: ' \
            '{}'.format(str(round(medftrscaler, 5)), str(round(medfvalscaler, 5)), str(round(medftescaler, 5)))
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig('/home/mah19006/projects/nonBDNA/data/median_forward.png')
    
    positions = np.arange(100)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # ax.plot(positions, medftr, linewidth=1, label='median forward training')
    # ax.plot(positions, medfval, linewidth=1, label='median forward validation')
    # ax.plot(positions, medfte, linewidth=1, label='median forward test')
    ax.plot(positions, medrtr, linewidth=1, label='Median Reverse Training')
    ax.plot(positions, medrval, linewidth=1, label='Median Reverse Validation')
    ax.plot(positions, medrte, linewidth=1, label='Median Reverse Test')
    plt.xlabel('Positions', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Value', fontsize=15)      # Add a y-label to the axes.
    ax.legend()
    title = 'Median in all positions' \
            '\nReverse Median TR: {}, Reverse Median val: {}, Reverse Median TE: ' \
            '{}'.format(str(round(medrtrscaler, 5)), str(round(medrvalscaler, 5)), str(round(medrtescaler, 5)))
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig('/home/mah19006/projects/nonBDNA/data/median_reverse.png')
    
    positions = np.arange(100)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(positions, iqrftr, linewidth=1, label='IQR Forward Training')
    ax.plot(positions, iqrfval, linewidth=1, label='IQR Forward Validation')
    ax.plot(positions, iqrfte, linewidth=1, label='IQR Forward Test')
    # ax.plot(positions, iqrrtr, linewidth=1, label='IQR reverse training')
    # ax.plot(positions, iqrrval, linewidth=1, label='IQR reverse validation')
    # ax.plot(positions, iqrrte, linewidth=1, label='IQR reverse test')
    plt.xlabel('Positions', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Value', fontsize=15)      # Add a y-label to the axes.
    ax.legend()
    title = 'IQR in all positions' \
            '\nForward IQR TR: {}, Forward IQR val: {}, Forward IQR TE: {}'.format(str(round(iqrftrscaler, 5)),
                                                                                   str(round(iqrfvalscaler, 5)),
                                                                                   str(round(iqrftescaler, 5)))
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig('/home/mah19006/projects/nonBDNA/data/IQR_forward.png')
    
    positions = np.arange(100)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # ax.plot(positions, iqrftr, linewidth=1, label='IQR forward training')
    # ax.plot(positions, iqrfval, linewidth=1, label='IQR forward validation')
    # ax.plot(positions, iqrfte, linewidth=1, label='IQR forward test')
    ax.plot(positions, iqrrtr, linewidth=1, label='IQR Reverse Training')
    ax.plot(positions, iqrrval, linewidth=1, label='IQR Reverse Validation')
    ax.plot(positions, iqrrte, linewidth=1, label='IQR Reverse Test')
    plt.xlabel('Positions', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Value', fontsize=15)  # Add a y-label to the axes.
    ax.legend()
    title = 'IQR in all positions' \
            '\nReverse IQR TR: {}, Reverse IQR val: {}, Reverse IQR TE: {}'.format(str(round(iqrrtrscaler, 5)),
                                                                                   str(round(iqrrvalscaler, 5)),
                                                                                   str(round(iqrrtescaler, 5)))
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig('/home/mah19006/projects/nonBDNA/data/IQR_reverse.png')




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
    to_print = 'if (!require("BiocManager", quietly = TRUE))\n' \
               '    install.packages("BiocManager")\n' \
               'BiocManager::install("IWTomics")\n' \
               'library(IWTomics)\n' \
               'data_path <- '+ data_path +'\n' \
                                           '\n' \
                                           '# make data:\n' \
                                           'datasets=read.table(file.path(' + data_path + ',"datasets.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)\n' \
                                                                                          'features_datasetsTable=read.table(file.path(' + data_path + ',"features_datasetsTable.txt"), sep="\t",header=TRUE,stringsAsFactors=FALSE)\n' \
                                                                                                                                                       'regionsFeatures=IWTomics::IWTomicsData(datasets$RegionFile,features_datasetsTable[,3:10],"center", datasets$id,datasets$name, features_datasetsTable$id,features_datasetsTable$name, path=file.path(' + data_path + ',"files"))\n' \
                                                                                                                                                                                                                                                                                                                                                                            'save(regionsFeatures,file=paste0(' + regions_path + '))\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                 '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                 '# Quantile test:\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                 'quantile_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c("A_Phased_Repeat", "G_Quadruplex_Motif", "Inverted_Repeat", "Mirror_Repeat", "Direct_Repeat", "Short_Tandem_Repeat","Z_DNA_Motif"), id_region2=c("Control", "Control", "Control", "Control", "Control", "Control", "Control"), statistics="quantile",probs=c(0.05,0.25,0.5,0.75,0.95),B=1000)\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                 'save(quantile_regionsFeatures_test_pairs,file=paste0(' + quantile_model_path +'))\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                '# Quantile test:\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'mean_regionsFeatures_test_pairs=IWTomicsTest(regionsFeatures,id_region1=c("A_Phased_Repeat", "G_Quadruplex_Motif", "Inverted_Repeat","Mirror_Repeat", "Direct_Repeat", "Short_Tandem_Repeat", "Z_DNA_Motif"), id_region2=c("Control", "Control", "Control", "Control", "Control", "Control", "Control"), statistics="median",B=1000)\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                'save(mean_regionsFeatures_test_pairs,file=paste0(' + median_model_path + '))\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          '\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          '# Adjusted p-value for median test:\n' \
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          'adjusted_pval(mean_regionsFeatures_test_pairs)\n'
    '\n' \
    '# Plots of median test:\n' \
    'plotTest(mean_regionsFeatures_test_pairs)\n' \
    'plotSummary(mean_regionsFeatures_test_pairs,groupby="feature",align_lab="Center")\n' \
    ''
    
    with open(os.path.join(data_path, 'iwt.R'), 'w') as f:
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
