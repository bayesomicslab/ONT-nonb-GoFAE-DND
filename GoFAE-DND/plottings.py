import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py
import os
from utils import prepare_bdna_dataset, collect_results
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




def plot_non_b_types_direction_quantiles_vs_control(path):
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


def plot_histogram(lst, name, save_path):
    plt.cla()
    plt.clf()
    plt.close()
    plt.hist(lst, bins=30)
    plt.xlabel(name)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, name + '.png'))


def plot_roc_curve(fpr, tpr, roc_auc_score, dataset, method, nonb, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=4, label='ROC Curve')
    plt.xlabel('False Positive Rate', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('True Negative Rate', fontsize=15)  # Add a y-label to the axes.
    ax.legend()
    title = '{} \nROC-AUC: {:.4f}'.format(method, roc_auc_score)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, dataset + '_' + method + '_' + nonb + '_eval_ROC.png'))


def plot_PR_curve(recall, precision, pr_auc, dataset, method, nonb, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(recall, precision, linewidth=4, label='PR Curve')
    plt.xlabel('Recall', fontsize=15)  # Add an x-label to the axes.
    plt.ylabel('Precision', fontsize=15)  # Add a y-label to the axes.
    ax.legend()
    title = '{}\nPR-AUC: {:.4f}'.format(method, pr_auc)
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, dataset + '_' + method + '_' + nonb + '_eval_PR.png'))


def plot_results_roc_pr(results_pd, plot_path, plot_type):
    """Plot roc or pr for sim data"""
    # path = 'results/sim_IF/final_results_sim_IF.csv'
    # results_pd = pd.read_csv(path, index_col=0)
    filtereted_results_pd = results_pd[results_pd['tail'] == 'upper'].reset_index(drop=True)
    train_sets = list(filtereted_results_pd['train_set'].unique())
    windows = list(filtereted_results_pd['window_size'].unique())
    # windows = [25, 50, 75, 100]
    non_bs = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
              'Short_Tandem_Repeat', 'Z_DNA_Motif']
    for win_id, win in enumerate(windows):
        for ts in train_sets:
            thisdf = filtereted_results_pd[(filtereted_results_pd['window_size'] == win) &
                                           (filtereted_results_pd['train_set'] == ts)].reset_index(drop=True)
            labels = list(thisdf['label'].unique())
            for label in labels:
                lb_df = thisdf[thisdf['label'] == label].reset_index(drop=True)
                nonb_ratios = list(lb_df['nonb_ratio'].unique())
                
                # if len(lb_df) < 19:
                #     print(win, ts, label)
                plot_name = ''
                x_labeb = ''
                y_labeb = ''
                fig, ax = plt.subplots(1, figsize=(7, 7))
                for nbr in nonb_ratios:
                    lb_nbratio = lb_df[lb_df['nonb_ratio'] == nbr].reset_index(drop=True)
                    if plot_type == 'ROC':
                        x = list(lb_nbratio['fpr'])
                        y = list(lb_nbratio['tpr'])
                        plot_name = '_'.join([ts, label, str(win)]) + '_ROC_plot.png'
                        x_labeb = 'FPR'
                        y_labeb = 'TPR'
                    elif plot_type == 'PR':
                        x = list(lb_nbratio['recall'])
                        y = list(lb_nbratio['precision'])
                        plot_name = '_'.join([ts, label, str(win)]) + '_PR_plot.png'
                        x_labeb = 'Recall'
                        y_labeb = 'Precision'
                    else:
                        return
                    # new_fpr, new_tpr = make_smooth(fpr, tpr, degree=3)
                    ax.plot(x, y, linewidth=3, label='Nonb ratio: ' + str(nbr))
                ax.set_xlabel(x_labeb, fontsize=13)  # Add an x-label to the axes.
                ax.set_ylabel(y_labeb, fontsize=13)  # Add a y-label to the axes.
                title = 'Window size: {}, \nTrained on: {}, \nTest on: {}'.format(win, ts, label)
                ax.set_title(title, fontsize=13)
                ax.legend(bbox_to_anchor=(1.1, 1.1))
                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(plot_path, plot_name))
    return

def plot_sim(results_path, results_name):
    results_sim_pd = collect_results(results_path, results_name)
    methods_results_path = os.path.join(results_path, results_name)
    plot_path = os.path.join(methods_results_path, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plot_results_roc_pr(results_sim_pd, plot_path, 'ROC')
    plot_results_roc_pr(results_sim_pd, plot_path, 'PR')

def aggregate_results_exp(main_path):
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


def aggregate_results_sim(main_path):
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


def plot_sim_comparison(path):
    
    df = aggregate_results_sim(path)
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


def plot_experimental_methods_comparison(path):
    exp_folders = [os.path.join(path, name) for name in os.listdir(path) if
                   'old' not in name and 'exp' in name and os.path.isdir(os.path.join(path, name))]
    nonbs = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
             'Short_Tandem_Repeat', 'Z_DNA_Motif']
    
    results_files = [os.path.join(f, elem, 'final_results.csv') for f in exp_folders for elem in os.listdir(f)
                     if os.path.isdir(os.path.join(f, elem)) and any(nonb in elem for nonb in nonbs)]
    results_pd_list = [pd.read_csv(file, index_col=0) for file in results_files if os.path.exists(file)]
    results_pd = pd.concat(results_pd_list, axis=0, ignore_index=True)
    plot_path = 'results/plots_exp'
    tail_types = ['upper', 'lower']
    
    for tail_type in tail_types:
        elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                    'Short_Tandem_Repeat', 'Z_DNA_Motif']
        elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                         'Short Tandem Repeat', 'Z-DNA']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
        
        # ['tab:brown', 'tab:gray', 'tab:olive', 'tab:cyan']
        results_pd = results_pd[results_pd['tail'] == 'upper'].reset_index(drop=True)
        for elem_id in range(len(elements)):
            elem_name = elements_name[elem_id]
            elem = elements[elem_id]
            df = results_pd[results_pd['label'] == elem].reset_index(drop=True)
            plt.cla()
            plt.close()
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            methods = list(df['method'].unique())
            # methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:orange', 'GoFAE-COD': 'tab:green',
            #                   'Outlier AE (64)': 'tab:red', 'Outlier AE (128)': 'tab:purple'}
            methods_colors = {'IF': 'tab:blue', 'LOF': 'tab:orange', 'SVM': 'tab:green'}
            for idx, method in enumerate(methods):
                this_df = df[df['method'] == method].reset_index(drop=True)
                tuples = list(zip(this_df.alpha, this_df.potential_nonb_counts))
                fdrs = [i[0] for i in tuples]
                n_nonb = [i[1] for i in tuples]
                color = methods_colors[method]
                ax.plot(fdrs, n_nonb, color=color, linewidth=3, marker='o', label=method)
                # ax.scatter(fdrs, n_nonb, color=color, marker='o', label=method)
            
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
            ax.set_title(elem_name + ', ' + tail_type, fontsize=20)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            plt.savefig(os.path.join(plot_path, 'exp_' + elem + '_' + tail_type + '.png'))


def plot_experimental_comparison_ismb(path):
    cols = ['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']
    
    IF_final = 'results/beagle2/exp_IF/final_results_exp_IF.csv'
    LOF_final = 'results/beagle2/exp_LOF/final_results_exp_LOF.csv'
    SVM_final = 'results/beagle2/exp_SVM/final_results_exp_SVM.csv'
    # GoFAE_DND_final = 'results/beagle2/exp_ourmodel/GoFAE-DND_final_results_0.35.csv'
    # AE_final = 'results/beagle/exp_AE/AE_final_results_exp.csv'
    
    if_df = pd.read_csv(IF_final, index_col=0)
    lof_df = pd.read_csv(LOF_final, index_col=0)
    svm_df = pd.read_csv(SVM_final, index_col=0)
    # gofae_df = pd.read_csv(GoFAE_DND_final, index_col=0)
    # ae_df = pd.read_csv(AE_final, index_col=0)
    
    if_df = if_df[if_df['tail'] == 'lower'].reset_index(drop=True)
    lof_df = lof_df[lof_df['tail'] == 'lower'].reset_index(drop=True)
    svm_df = svm_df[svm_df['tail'] == 'lower'].reset_index(drop=True)
    if_df = if_df.drop(['training_time', 'tail'], axis=1)
    lof_df = lof_df.drop(['training_time', 'tail'], axis=1)
    svm_df = svm_df.drop(['training_time', 'tail'], axis=1)
    
    # gofae_df['dataset'] = 'experimental'
    # gofae_df['method'] = 'GoFAE-DND'
    # gofae_df['dataset'] = 'experimental'
    # ae_df['dataset'] = 'experimental'
    
    if_df['method'] = 'Isolation Forest'
    lof_df['method'] = 'Local Outlier Factor'
    svm_df['method'] = 'One Class SVM'
    
    if_df = if_df[cols]
    lof_df = lof_df[cols]
    # gofae_df = gofae_df[cols]
    svm_df = svm_df[cols]
    # ae_df = ae_df[cols]
    
    # gofae_df = gofae_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    # if_df = if_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    # lof_df = lof_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    # svm_df = svm_df[['dataset', 'method', 'label', 'alpha', 'potential_nonb_counts']]
    
    # all_results = pd.concat([if_df, lof_df, svm_df, gofae_df, ae_df], ignore_index=True)
    all_results = pd.concat([if_df, lof_df, svm_df], ignore_index=True)
    all_results = all_results[all_results['alpha'] <= 0.5]
    all_results.to_csv('results/beagle2/all_experimental.csv')
    plot_path = 'results//beagle2/plots_exp'
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    elements = ['A_Phased_Repeat', 'G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Direct_Repeat',
                'Short_Tandem_Repeat', 'Z_DNA_Motif']
    elements_name = ['A Phased Repeat', 'G-Quadruplex', 'Inverted Repeat', 'Mirror Repeat', 'Direct Repeat',
                     'Short Tandem Repeat', 'Z-DNA']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    methods_colors = {'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:orange', 'GoFAE-DND': 'tab:green',
                      'AE': 'tab:red', 'Outlier AE (128)': 'tab:purple', 'Local Outlier Factor': 'tab:pink'}
    markers_markers = {'Isolation Forest': 'd', 'One Class SVM': '*', 'GoFAE-DND': 'o',
                       'AE': '|', 'Outlier AE (128)': 'x', 'Local Outlier Factor': 'P'}
    
    for elem_id in range(len(elements)):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        df = all_results[all_results['label'] == elem].reset_index(drop=True)
        plt.cla()
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        methods = list(df['method'].unique())
        
        for idx, method in enumerate(methods):
            this_df = df[df['method'] == method].reset_index(drop=True)
            tuples = list(zip(this_df.alpha, this_df.potential_nonb_counts))
            fdrs = [i[0] for i in tuples]
            n_nonb = [i[1] for i in tuples]
            color = methods_colors[method]
            marker = markers_markers[method]
            ax.plot(fdrs, n_nonb, color=color, linewidth=3, marker=marker, label=method, markersize=10, alpha=0.7)
            # ax.scatter(fdrs, n_nonb, color=color, marker='o', label=method)
        
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
        ax.set_title(elem_name, fontsize=20)
        plt.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(plot_path, 'exp_' + elem + '.png'))


def plot_classifiers_ismb1():
    classifiers_path = 'results/classifiers'


def plot_roc(results_sim_folder):
    out_liers = ['IF', 'LOF', 'SVM']
    classifiers = ['SVC', 'RF', 'GP', 'KNN', 'LR']
    # outfiles = [os.path.join(results_sim, name, 'final_results_' + name + '.csv') for name in os.listdir(results_sim)
    #            if 'sim' in name and any(ol in name for ol in out_liers)]
    needed_cols = ['dataset', 'method', 'label', 'nonb_ratio', 'fscore']
    needed_nonb_ratios = [0.05, 0.1, 0.25]
    classifiles = [os.path.join(results_sim_folder, name, 'final_results_' + name + '.csv') for name in os.listdir(results_sim_folder)
                   if 'sim' in name and any(cl in name for cl in classifiers)]
    cls = [pd.read_csv(f, index_col=0) for f in classifiles]
    all_classifiers = pd.concat(cls, ignore_index=True)


def find_mask_and_genomic_positions_in_100_wins(win_start, win_end, motif):
    ml = len(motif)
    mask = np.ones(100)
    if ml % 2 == 0:
        a = ml / 2
        b = 50 - a
        motif_start = win_start + b
        motif_end = win_end - b
        mask[0:int(b)] = 0
        mask[-int(b):] = 0
    # ml%2 == 1: # motif length is odd
    else:
        a = np.floor(ml / 2)
        b_start = 50 - (a + 1)
        b_end = 50 - a
        motif_start = win_start + b_start
        motif_end = win_end - b_end
        mask[0:int(b_start)] = 0
        mask[-int(b_end):] = 0
    return motif_start, motif_end, mask


def prepare_sim_all_results_ismb():
    alpha = 0.2
    results_sim = 'results'
    plot_path = os.path.join(results_sim, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    out_liers = ['IF', 'LOF', 'SVM']
    classifiers = ['SVC', 'RF', 'GP', 'KNN', 'LR']
    elements = ['G_Quadruplex_Motif', 'Short_Tandem_Repeat']
    elements_name = ['G Quadruplex', 'Short Tandem Repeat']
    methods_colors = {'GoFAE-DND': 'tab:red', 'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:green',
                      'Local Outlier Factor': 'tab:orange',
                      'Isolation Forest (bdna)': '#acc2d9', 'One Class SVM (bdna)': 'black',
                      'Local Outlier Factor (bdna)': '#388004', 'GP': 'tab:purple', 'KNN': '#DBB40C',
                      'LR': 'tab:cyan', 'RF': 'tab:brown', 'SVC': 'tab:pink'}
    
    method_order = {'GoFAE-DND': 0, 'Isolation Forest': 1, 'Isolation Forest (bdna)': 2, 'Local Outlier Factor': 3,
                    'Local Outlier Factor (bdna)': 4, 'One Class SVM': 5, 'One Class SVM (bdna)': 6,
                    'KNN': 7, 'LR': 8, 'RF': 9, 'SVC': 10, 'GP': 11}
    # outlier_order = {'GoFAE-DND': 0, 'Isolation Forest': 1, 'Isolation Forest (bdna)': 2, 'Local Outlier Factor': 3,
    #                  'Local Outlier Factor (bdna)': 4, 'One Class SVM': 5, 'One Class SVM (bdna)': 6}
    #
    # classifiers_order = {'KNN': 7, 'LR': 8, 'RF': 9, 'SVC': 10, 'GP': 11}
    
    needed_cols = ['dataset', 'method', 'label', 'nonb_ratio', 'alpha', 'fscore']
    needed_cols_classifiers = ['dataset', 'method', 'label', 'nonb_ratio', 'fscore']
    needed_nonb_ratios = [0.05, 0.1, 0.25]
    classifiles = [os.path.join(results_sim, name, 'final_results_' + name + '.csv') for name in os.listdir(results_sim)
                   if 'sim' in name and any(cl in name for cl in classifiers)]
    cls = [pd.read_csv(f, index_col=0) for f in classifiles]
    all_classifiers = pd.concat(cls, ignore_index=True)
    all_classifiers = all_classifiers[needed_cols_classifiers]
    all_classifiers = all_classifiers[all_classifiers['nonb_ratio'].isin(needed_nonb_ratios)].reset_index(drop=True)
    all_classifiers.to_csv(os.path.join(results_sim, 'simulation_results_classifiers_methods.csv'))
    
    IF_final = os.path.join(results_sim, 'sim_IF/final_results_sim_IF.csv')
    LOF_final = os.path.join(results_sim, 'sim_LOF/final_results_sim_LOF.csv')
    SVM_final = os.path.join(results_sim, 'sim_SVM/final_results_sim_SVM.csv')
    IF_bdna_final = os.path.join(results_sim, 'sim_IF_bdna/final_results_sim_IF_bdna.csv')
    LOF_bdna_final = os.path.join(results_sim, 'sim_LOF_bdna/final_results_sim_LOF_bdna.csv')
    SVM_bdna_final = os.path.join(results_sim, 'sim_SVM_bdna/final_results_sim_SVM_bdna.csv')
    all_paths = [IF_final, LOF_final, SVM_final, IF_bdna_final, LOF_bdna_final, SVM_bdna_final]
    all_dfs = [pd.read_csv(path, index_col=0) for path in all_paths]
    dfs = pd.concat(all_dfs).reset_index(drop=True)
    dfs = dfs[dfs['tail'] == 'lower'].reset_index(drop=True)
    dfs = dfs[dfs['function'] == 'decision_function'].reset_index(drop=True)
    dfs = dfs.drop(['duration', 'tail', 'function'], axis=1)
    dfs.loc[dfs['method'] == 'IF', 'method'] = 'Isolation Forest'
    dfs.loc[dfs['method'] == 'LOF', 'method'] = 'Local Outlier Factor'
    dfs.loc[dfs['method'] == 'SVM', 'method'] = 'One Class SVM'
    dfs.loc[dfs['method'] == 'IF_bdna', 'method'] = 'Isolation Forest (bdna)'
    dfs.loc[dfs['method'] == 'LOF_bdna', 'method'] = 'Local Outlier Factor (bdna)'
    dfs.loc[dfs['method'] == 'SVM_bdna', 'method'] = 'One Class SVM (bdna)'
    dfs = dfs[needed_cols]
    
    gofae_path = os.path.join(results_sim, 'sim_GoFAE-DND')
    gofae_files = [pd.read_csv(os.path.join(gofae_path, name), index_col=0) for name in os.listdir(gofae_path) if
                   '.csv' in name]
    gofae_df = pd.concat(gofae_files, ignore_index=True)
    gofae_df = gofae_df.rename(columns={'nonb_type': 'label'})
    gofae_df = gofae_df.drop(['upper'], axis=1)
    gofae_df = gofae_df[needed_cols]
    
    all_out = pd.concat([dfs, gofae_df], ignore_index=True)
    all_out = all_out[(all_out['alpha'] > alpha - 0.01) & (all_out['alpha'] < alpha + 0.01)].reset_index(drop=True)
    
    all_out = all_out.drop(['alpha'], axis=1)
    all_out = all_out[needed_cols_classifiers]
    all_out['dataset'] = 'sim'
    all_out = all_out.sort_values(by=['dataset', 'method', 'label', 'nonb_ratio'])
    all_out = all_out[all_out['nonb_ratio'].isin(needed_nonb_ratios)].reset_index(drop=True)
    # all_out = all_out.fillna(0)
    # all_out = all_out.dropna()
    all_out.to_csv(os.path.join(results_sim, 'simulation_results_outlier_methods.csv'))
    everything = pd.concat([all_out, all_classifiers], ignore_index=True)
    everything = everything[everything['nonb_ratio'].isin(needed_nonb_ratios)].reset_index(drop=True)
    everything = everything.sort_values(by=['dataset', 'method', 'label', 'nonb_ratio']).reset_index(drop=True)
    everything.to_csv(os.path.join(results_sim, 'all_simulation_results.csv'))
    
    everything['method_order'] = everything['method'].apply(lambda x: method_order[x])
    everything = everything.sort_values(by='method_order', ascending=True)
    everything = everything.drop(['method_order'], axis=1).reset_index(drop=True)
    
    nonb_ratios = [0.05, 0.1, 0.25]
    
    outlier_order = {'Isolation Forest': 0, 'Local Outlier Factor': 1, 'One Class SVM': 2, 'GoFAE-DND': 3}
    classifiers_order = {'KNN': 4, 'GP': 5, 'RF': 6, 'LR': 7, 'SVC': 8}
    barWidth = 0.4
    for elem_id in range(2):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        plt.cla()
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        x = 0
        ax.axhline(0.1, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.2, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.3, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.4, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.5, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.6, color="gray", linestyle='--', zorder=0)
        # ax.axhline(0.7, color="gray", linestyle='--', zorder=0)
        # ax.axhline(0.8, color="gray", linestyle='--', zorder=0)
        for nbr in nonb_ratios[:-1]:
            
            for me in list(classifiers_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, hatch='//', edgecolor='black')
                x += 0.5
            
            x += 0.5
            
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black')
                x += 0.5
            
            x += 2
        
        for nbr in [nonb_ratios[-1]]:
            
            for me in list(classifiers_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black', hatch='//', label=me)
                x += 0.5
            
            x += 0.5
            
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black', label=me)
                x += 0.5
        
        # leg = plt.legend(loc=(1.03, 0.50), title="Quantile", prop={'size': 14}, title_fontsize=15)
        # ax.add_artist(leg)
        # h = [plt.plot([], [], color="gray", linestyle=ls, linewidth=3)[0] for ls in ['-', '--']]
        # plt.legend(handles=h, labels=['Non-B DNA', 'Control'], loc=(1.03, 0.85),
        #            title="Structure", prop={'size': 14}, title_fontsize=15)
        
        # plt.xlabel('Non-B ratios', fontsize=27, labelpad=30)
        plt.xlabel('Non-B ratios', fontsize=27)
        plt.xticks([x / 6 - 1, x / 2, 5 * x / 6 + 0.5], [str(i) for i in nonb_ratios])
        plt.ylabel('F1 Score', fontsize=27)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        # plt.legend(labels = ['KNN', 'GP', 'RF', 'LR', 'SVC'], loc=(1.03, 0.2), title="Classifiers",
        #            prop={'size': 14}, title_fontsize=15)
        # plt.legend(labels = ['Isolation Forest', 'Local Outlier Factor', 'One Class SVM', 'GoFAE-DND'],
        #            loc=(1.03, 0.2), title="Novelty Detectors",
        #            prop={'size': 14}, title_fontsize=15)
        
        plt.tight_layout()
        # plt.savefig(os.path.join(plot_path, elem + 'F1_score.tiff'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score.png'), dpi=1200)
        # plt.savefig(os.path.join(plot_path, elem + 'F1_score.tif'), dpi=1200)
        # plt.savefig(os.path.join(plot_path, elem + 'F1_score.pdf'), dpi=1200)
        
        # plt.savefig(os.path.join(plot_path, 'leg2.png'), dpi=1200)


def prepare_sim_outliers_results_ismb():
    alpha = 0.2
    results_sim = 'results'
    plot_path = os.path.join(results_sim, 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    out_liers = ['IF', 'LOF', 'SVM']
    elements = ['G_Quadruplex_Motif', 'Short_Tandem_Repeat']
    elements_name = ['G Quadruplex', 'Short Tandem Repeat']
    methods_colors = {'GoFAE-DND': 'tab:red', 'Isolation Forest': 'tab:blue', 'One Class SVM': 'tab:green',
                      'Local Outlier Factor': 'tab:orange',
                      'Isolation Forest (bdna)': '#acc2d9', 'One Class SVM (bdna)': 'black',
                      'Local Outlier Factor (bdna)': '#388004', 'GP': 'tab:purple', 'KNN': '#DBB40C',
                      'LR': 'tab:cyan', 'RF': 'tab:brown', 'SVC': 'tab:pink'}
    
    outlier_order = {'Isolation Forest': 0, 'Isolation Forest (bdna)': 1, 'Local Outlier Factor': 2, 'GoFAE-DND': 3,
                     'Local Outlier Factor (bdna)': 4, 'One Class SVM': 5, 'One Class SVM (bdna)': 6}
    
    needed_cols = ['dataset', 'method', 'label', 'nonb_ratio', 'alpha', 'fscore']
    
    outlier_order = {'Isolation Forest': 0, 'Local Outlier Factor': 1, 'One Class SVM': 2, 'GoFAE-DND': 3}
    classifiers_order = {'KNN': 4, 'LR': 5, 'RF': 6, 'SVC': 7, 'GP': 8}
    
    nonb_ratios = [0.05, 0.1, 0.25]
    
    IF_final = os.path.join(results_sim, 'sim_IF/final_results_sim_IF.csv')
    LOF_final = os.path.join(results_sim, 'sim_LOF/final_results_sim_LOF.csv')
    SVM_final = os.path.join(results_sim, 'sim_SVM/final_results_sim_SVM.csv')
    IF_bdna_final = os.path.join(results_sim, 'sim_IF_bdna/final_results_sim_IF_bdna.csv')
    LOF_bdna_final = os.path.join(results_sim, 'sim_LOF_bdna/final_results_sim_LOF_bdna.csv')
    SVM_bdna_final = os.path.join(results_sim, 'sim_SVM_bdna/final_results_sim_SVM_bdna.csv')
    all_paths = [IF_final, LOF_final, SVM_final, IF_bdna_final, LOF_bdna_final, SVM_bdna_final]
    all_dfs = [pd.read_csv(path, index_col=0) for path in all_paths]
    dfs = pd.concat(all_dfs).reset_index(drop=True)
    dfs = dfs[dfs['tail'] == 'lower'].reset_index(drop=True)
    dfs = dfs[dfs['function'] == 'decision_function'].reset_index(drop=True)
    dfs = dfs.drop(['duration', 'tail', 'function'], axis=1)
    dfs.loc[dfs['method'] == 'IF', 'method'] = 'Isolation Forest'
    dfs.loc[dfs['method'] == 'LOF', 'method'] = 'Local Outlier Factor'
    dfs.loc[dfs['method'] == 'SVM', 'method'] = 'One Class SVM'
    dfs.loc[dfs['method'] == 'IF_bdna', 'method'] = 'Isolation Forest (bdna)'
    dfs.loc[dfs['method'] == 'LOF_bdna', 'method'] = 'Local Outlier Factor (bdna)'
    dfs.loc[dfs['method'] == 'SVM_bdna', 'method'] = 'One Class SVM (bdna)'
    dfs = dfs[needed_cols]
    
    gofae_path = os.path.join(results_sim, 'sim_GoFAE-DND')
    gofae_files = [pd.read_csv(os.path.join(gofae_path, name), index_col=0) for name in os.listdir(gofae_path) if
                   '.csv' in name]
    gofae_df = pd.concat(gofae_files, ignore_index=True)
    gofae_df = gofae_df.rename(columns={'nonb_type': 'label'})
    gofae_df = gofae_df.drop(['upper'], axis=1)
    gofae_df = gofae_df[needed_cols]
    
    all_out = pd.concat([dfs, gofae_df], ignore_index=True)
    all_out = all_out[(all_out['alpha'] > alpha - 0.01) & (all_out['alpha'] < alpha + 0.01)].reset_index(drop=True)
    # all_out = all_out.drop(['alpha'], axis=1)
    all_out['dataset'] = 'sim'
    all_out = all_out.sort_values(by=['dataset', 'method', 'label', 'nonb_ratio']).reset_index(drop=True)
    all_out.to_csv(os.path.join(results_sim, 'simulation_results_outlier_methods.csv'))
    everything = all_out
    
    barWidth = 0.3
    for elem_id in range(2):
        elem_name = elements_name[elem_id]
        elem = elements[elem_id]
        plt.cla()
        plt.close()
        plt.clf()
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        x = 0
        ax.axhline(0.1, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.2, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.3, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.4, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.5, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.6, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.7, color="gray", linestyle='--', zorder=0)
        ax.axhline(0.8, color="gray", linestyle='--', zorder=0)
        # ax.axhline(0.9, color="gray", linestyle='--', zorder=0)
        
        for nbr in nonb_ratios[:-1]:
            print(nbr)
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                print(nbr, me, y)
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black')
                x += 0.3
            
            x += 0.6
        
        for nbr in [nonb_ratios[-1]]:
            print(nbr)
            for me in list(outlier_order.keys()):
                this_df = everything[(everything['method'] == me) & (everything['label'] == elem) & (
                        everything['nonb_ratio'] == nbr)].reset_index(drop=True)
                y = this_df.fscore.values[0]
                print(nbr, me, y)
                plt.bar(x, y, color=methods_colors[me], width=barWidth, edgecolor='black', label=me)
                x += 0.3
        
        # plt.xlabel('Non-B ratios', fontsize=27, labelpad=30)
        plt.xlabel('Non-B ratios', fontsize=27)
        # plt.xlabel("...", labelpad=20)
        plt.xticks([2 * 0.3 - 0.1, 8 * 0.3 - 0.1, 14 * 0.3 - 0.1], [str(i) for i in nonb_ratios])
        plt.ylabel('F1 Score', fontsize=27)
        plt.setp(ax.get_xticklabels(), Fontsize=22)
        plt.setp(ax.get_yticklabels(), Fontsize=22)
        plt.legend(loc=2, prop={'size': 17})
        # plt.annotate('Outliers', xy=(0, 2), xycoords='data', xytext=(1.5, 1.5),
        #              textcoords='offset points')
        plt.tight_layout()
        
        # plt.show()
        # plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.tiff'), dpi=1200)
        plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.png'), dpi=1200)
        # plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.tif'), dpi=1200)
        # plt.savefig(os.path.join(plot_path, elem + 'F1_score_outliers.pdf'), dpi=1200)


if __name__ == '__main__':
    prepare_sim_outliers_results_ismb()
    prepare_sim_all_results_ismb()