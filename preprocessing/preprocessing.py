import sys
import os
import networkx as nx
import pandas as pd
import numpy as np
from multiprocessing import Pool
from utils import make_reverse_complete
from subprocess import call

def get_interval(row):
    if row.motif_len > 100:
        biger_interval = (row.start, row.end)
    else:
        biger_interval = (row.win_start, row.win_end)
    return biger_interval


def gff2bed_all_chr():
    """Note gff file is indexed starting from 1 (the first position on genome is one)
    Input: nonb dna database gff file (1-based)
    Output: Dataframe of necessary information (0-based)"""
    main_folder = '/labs/Aguiar/non_bdna/annotations/human_hg38.gff'
    gff_files = [name for name in os.listdir(main_folder) if '.gff' in name and 'chrMT' not in name]
    add_non_b_ann = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand'])
    for gf in gff_files:
        gff_df = pd.read_csv(os.path.join(main_folder, gf), names=['chr', 'source', 'feature', 'start', 'end', 'score',
                                                                   'strand', 'frame', 'attribute'], skiprows=[0],
                            sep='\t')
        gff_df['motif_id'] = gff_df['attribute'].apply(lambda x: x.split(';')[0].split('=')[1])
        seq_idx = [idx for idx, s in enumerate(gff_df.loc[0, 'attribute'].split(';')) if 'sequence' in s][0]
        gff_df['motif_seq'] = gff_df['attribute'].apply(lambda x: x.split(';')[seq_idx].split('=')[1])
        gff_df['motif_len'] = gff_df.apply(lambda row: row.end - row.start + 1, axis=1)
        gff_df['start'] = gff_df['start'] - 1
        gff_df['end'] = gff_df['end'] - 1
        gff_df = gff_df.drop(columns=['source', 'score', 'frame', 'attribute'])
        gff_df = gff_df.dropna()
        add_non_b_ann = add_non_b_ann.append(gff_df, ignore_index=True).reset_index(drop=True)
    add_non_b_ann.to_csv('/labs/Aguiar/non_bdna/annotations/all_non_b_motifs_0_based.csv', index=False)


def fix_windows_on_motifs(ms, me):
    """ms: motif start, me: motif end"""
    ml = me - ms + 1
    # motif length is even
    if ml % 2 == 0:
        a = ml/2
        b = 50 - a
        win_s = ms-b
        win_e = me+b
    # ml%2 == 1: # motif length is odd
    else:
        a = np.floor(ml/2)
        b_start = 50 - (a + 1)
        b_end = 50 - a
        win_s = ms - b_start
        win_e = me + b_end
    return [win_s, win_e]



def make_windows_on_motifs():
    add_non_b_ann = pd.read_csv('/labs/Aguiar/non_bdna/annotations/all_non_b_motifs_0_based.csv')
    add_non_b_ann['win_start'] = add_non_b_ann.apply(lambda row: fix_windows_on_motifs(row.start, row.end)[0], axis=1)
    add_non_b_ann['win_end'] = add_non_b_ann.apply(lambda row: fix_windows_on_motifs(row.start, row.end)[1], axis=1)
    convert_dict = {'win_start': int, 'win_end': int}
    add_non_b_ann = add_non_b_ann.astype(convert_dict)
    add_non_b_ann.to_csv('/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based.csv', index=False)


def get_opposite_direction(row):
    opposite_strand = {'+': '-', '-': '+'}
    chr, feature, start, end, strand, motif_id, motif_seq, motif_len, win_start, win_end, direction = row
    # row['chr'] = chr
    # row['feature'] = feature
    # row['start'] = start
    # row['end'] = end
    # strand = row['strand']
    row['strand'] = opposite_strand[strand]
    # row['motif_id'] = motif_id
    # motif_seq = row['motif_seq']
    row['motif_seq'] = make_reverse_complete(motif_seq)
    # row['motif_len'] = motif_len
    # row['win_start'] = win_start
    # row['win_end'] = win_end
    row['direction'] = 'opposite'
    return row

def extend_windows_to_other_strand_direction():
    """This function reads the windows and computes the windows on the other strand.
    The sequence should be reverse complement"""
    # main_folder = '/labs/Aguiar/non_bdna/annotations/' # UCHC
    main_folder = '/home/mah19006/projects/nonBDNA/data/' # beagle
    # main_folder = 'Data/windows/' # local
    path = os.path.join(main_folder, 'all_non_b_windows_0_based.csv')
    save_path = os.path.join(main_folder, 'all_non_b_windows_0_based_both_directions.csv')
    non_b_0 = pd.read_csv(path)
    convert_dict = {'motif_len': int}
    non_b_0 = non_b_0.astype(convert_dict)
    non_b_0['direction'] = 'same'
    non_b_0_opposite = non_b_0.apply(lambda row: get_opposite_direction(row), axis=1)
    non_b_0_opposite.to_csv('all_non_b_windows_0_based_opposite.csv', index=False)
    non_b_0_both_dir = non_b_0.append(non_b_0_opposite).reset_index(drop=True)
    non_b_0_both_dir = non_b_0_both_dir.sort_values(by=['feature', 'win_start']).reset_index(drop=True)
    non_b_0_both_dir.to_csv('all_non_b_windows_0_based_both_directions.csv', index=False)
    

def make_query_bounds_per_chr():
    # add_non_b_ann = pd.read_csv('/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based_both_directions.csv') # UCHC
    add_non_b_ann = pd.read_csv('/mnt/research/aguiarlab/proj/nonBDNA/data/all_non_b_windows_0_based_both_directions.csv') # beagle
    # mapping_file_path = '/labs/Aguiar/non_bdna/annotations/mapping_chr_chromosome.csv' # UCHC
    mapping_file_path = '/mnt/research/aguiarlab/proj/nonBDNA/data/mapping_chr_chromosome.csv' # beagle
    mapping_file = pd.read_csv(mapping_file_path, index_col=False)
    chr_dict = {mapping_file.loc[i, 'chr']: mapping_file.loc[i, 'chromosome'] for i in range(len(mapping_file))}
    non_b_g_min = add_non_b_ann.groupby(['chr', 'strand'])['win_start'].min()
    non_b_g_max = add_non_b_ann.groupby(['chr', 'strand'])['win_end'].max()
    groups = list(non_b_g_min.index)
    queries_df = pd.DataFrame(columns=['chr', 'strand', 'start', 'end'], index=range(len(groups)))
    
    i = 0
    for gr in groups:
        chrom = gr[0]
        strand = gr[1]
        start = non_b_g_min.loc[gr]
        end = non_b_g_max.loc[gr]
        queries_df.loc[i, :] = chrom, strand, start, end
        i += 1
    queries_df['chromosome'] = queries_df['chr'].apply(lambda x: chr_dict[x])
    queries_df['start'] = queries_df['start'] + 1
    queries_df['end'] = queries_df['end'] + 1
    # queries_df.to_csv('/labs/Aguiar/non_bdna/annotations/chr_start_end_queries_1_based.csv', index=False) # UCHC
    queries_df.to_csv('/mnt/research/aguiarlab/proj/nonBDNA/data/chr_start_end_queries_1_based.csv', index=False) # beagle


def filter_query_samtools():
    save_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/filtered_per_chr/'
    bam_path = '/labs/Aguiar/non_bdna/human/na12878/bam_files/merged.bam'
    # path = '/mnt/research/aguiarlab/proj/nonBDNA/data/chr_start_end_queries_1_based.csv' # beagle
    path = '/labs/Aguiar/non_bdna/annotations/chr_start_end_queries_1_based.csv'
    df = pd.read_csv(path, index_col=False)
    sh_path = '/labs/Aguiar/non_bdna/human/scripts/lookup_paper2.sh'
    to_print = '#!/bin/bash\n' \
                '#BATCH --job-name=signal\n' \
                '#SBATCH -N 1\n' \
                '#SBATCH -n 1\n' \
                '#SBATCH -c 1\n' \
                '#SBATCH --partition=general\n' \
                '#SBATCH --qos=general\n' \
                '#SBATCH --mail-type=END\n' \
                '#SBATCH --mem=10G\n' \
                '#SBATCH --mail-user=marjan.hosseini@uconn.edu\n' \
                '#SBATCH -o query_paper2.out\n' \
                '#SBATCH -e query_paper2.err\n' \
                '\n' \
                'echo `hostname`\n' \
                'module load samtools\n' \
                'module load bedtools\n' \
                '\n'
    for i in range(len(df)):
        chr = df.loc[i, 'chromosome']
        strand = df.loc[i, 'strand']
        chrno = df.loc[i, 'chr']
        start = df.loc[i, 'start']
        end = df.loc[i, 'end']
        name = chrno + strand
        query = chr + ':' + str(start) + '-' + str(end)
        comm1 = ''
        if strand == '+':
            comm1 = 'samtools view {} {} -F 0xF14 -o {}{}.bam;'.format(bam_path, query, save_path, name)
        elif strand == '-':
            comm1 = 'samtools view {} {} -F 0xF04 -f 0x10 -o {}{}.bam;'.format(bam_path, query, save_path, name)
        comm2 = 'bedtools bamtobed -i {}{}.bam > {}{}.bed;'.format(save_path, name, save_path, name)
        print(comm1)
        print(comm2)
        to_print += comm1 + '\n'
        to_print += comm2 + '\n'
    with open(sh_path, 'w') as f:
        f.write(to_print)
    # run the command
    call(sh_path, shell=True)
    

def make_motif_free_intervals():
    # path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based.csv'
    path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based_both_directions.csv'
    windows = pd.read_csv(path)
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/control_conservative_0_based/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    chr_strand_windows_inputs = []
    non_b_groups = windows.groupby(['chr', 'strand']).count()
    groups = list(non_b_groups.index)
    for gr in groups:
        chr, stra = gr
        # print(chr, stra)
        this_ch_stra = windows[(windows['chr'] == chr) & (windows['strand'] == stra)].reset_index(drop=True)
        print(chr, stra, len(this_ch_stra))
        chr_strand_windows_inputs.append([chr, stra, this_ch_stra, save_path])

    pool = Pool(40)
    # pool.map(make_control_windows_chr_strand, chr_strand_windows_inputs)
    pool.map(make_control_windows_chr_strand_conservative, chr_strand_windows_inputs)

    control_windows = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand', 'motif_len', 'win_start',
                                            'win_end'])
    control_files = [name for name in os.listdir(save_path) if '.csv' in name]
    for cf in control_files:
        cf_df = pd.read_csv(os.path.join(save_path, cf), index_col=0)
        control_windows = control_windows.append(cf_df, ignore_index=True)
    control_windows.to_csv(os.path.join(save_path, 'all_controls_conservative_0_based.csv'))


def make_control_windows_chr_strand_conservative(inp):
    # conservative
    chr, stra, this_ch_stra, save_path = inp

    motif_intervals_list = list(zip(this_ch_stra.start, this_ch_stra.end))
    windows_intervals_list = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    interval_list = motif_intervals_list + windows_intervals_list

    # find unions of regions
    union = []
    for begin, end in sorted(interval_list):
        if union and union[-1][1] >= begin - 1:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([begin, end])

    union = sorted(union, key=lambda x: x[1])
    # find between regions
    motif_free_intervals = []
    for i in range(len(union) - 1):
        free = [union[i][1] + 1, union[i + 1][0] - 1]
        if free[1] - free[0] > 140:
            # print(free[1] - free[0])
            motif_free_intervals.append(free)

    # make the windows (bed file)
    control_df = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand', 'motif_len', 'win_start', 'win_end'],
                              index=range(len(motif_free_intervals)))
    # tt = []
    for i in range(len(motif_free_intervals)):
        start = motif_free_intervals[i][0]
        end = motif_free_intervals[i][1]
        mid = int(np.floor((start + end) / 2))
        win_start = mid - 49
        win_end = mid + 50
        # tt.append((win_start, win_end))
        # print(motif_free_intervals[i],'==>',(win_start, win_end))
        control_df.loc[i, :] = chr, 'Control', win_start, win_end, stra, 100, win_start, win_end

    control_df.to_csv(os.path.join(save_path, 'control_conservative_' + chr + stra + '.csv'))


def find_non_overlapping_windows():
    
    # path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based.csv'
    path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based_both_directions.csv'
    
    # control_path = '/labs/Aguiar/non_bdna/annotations/all_control_windows_conservative_0_based.csv'
    control_path = '/labs/Aguiar/non_bdna/annotations/windows/control_conservative_0_based/all_controls_conservative_0_based.csv'
    
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping_0_based'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    windows = pd.read_csv(path)
    windows['motif_len'] = windows['motif_len'].astype('int')
    controls = pd.read_csv(control_path, index_col=0)

    all_windows = windows.append(controls, ignore_index=True).reset_index(drop=True)
    all_windows.loc[((all_windows['feature'] == 'Control') & (all_windows['strand'] == '+')), 'direction'] = 'same'
    all_windows.loc[((all_windows['feature'] == 'Control') & (all_windows['strand'] == '-')), 'direction'] = 'opposite'
    all_windows.loc[(all_windows['feature'] == 'Control'), 'motif_seq'] = ''
    all_windows.loc[(all_windows['feature'] == 'Control'), 'motif_id'] = ''
    # all_windows = windows
    # print(chr, stra)
    # this_ch_stra = all_windows[(all_windows['chr'] == chrom) & (all_windows['strand'] == strand)].reset_index(drop=True)
    inputs = []
    chromosomes = sorted(list(all_windows['chr'].unique()))
    # strands = ['+', '-']
    for chrom in chromosomes:
        # for strand in strands:
        # this_ch_stra = all_windows[(all_windows['chr'] == chrom) & (all_windows['strand'] == '+') &
        # (all_windows['feature'] != 'Control')].reset_index(drop=True)
        strand = '+'
        this_ch_stra = all_windows[(all_windows['chr'] == chrom) & (all_windows['strand'] == strand)].reset_index(drop=True)
        print(chrom, strand, len(this_ch_stra))
        inputs.append([chrom, strand, this_ch_stra, save_path])
    del all_windows
    # find_non_overlapping_windows_pre_chr_distributed(chrom, strand, this_ch_stra, save_path, threads)
    pool = Pool(24)
    pool.map(find_non_overlapping_windows_pre_chr_no_graph, inputs)
    

def old_find_non_overlapping_windows():
    # path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based.csv'
    path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows_0_based_both_directions.csv'
    
    # control_path = '/labs/Aguiar/non_bdna/annotations/all_control_windows_conservative_0_based.csv'
    control_path = '/labs/Aguiar/non_bdna/annotations/windows/control_conservative_0_based/all_controls_conservative_0_based.csv'
    
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping_0_based'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    windows = pd.read_csv(path)
    windows['motif_len'] = windows['motif_len'].astype('int')
    # controls = pd.read_csv(control_path, index_col=0)
    
    # all_windows = windows.append(controls, ignore_index=True).reset_index(drop=True)
    all_windows = windows
    # print(chr, stra)
    # this_ch_stra = all_windows[(all_windows['chr'] == chrom) & (all_windows['strand'] == strand)].reset_index(drop=True)
    inputs = []
    chromosomes = sorted(list(all_windows['chr'].unique()))
    strands = ['+', '-']
    for chrom in chromosomes:
        for strand in strands:
            this_ch_stra = all_windows[(all_windows['chr'] == chrom) & (all_windows['strand'] == strand)].reset_index(drop=True)
            print(chrom, strand, len(this_ch_stra))
            inputs.append([chrom, strand, this_ch_stra, save_path])
    
    del all_windows
    # find_non_overlapping_windows_pre_chr_distributed(chrom, strand, this_ch_stra, save_path, threads)
    pool = Pool(30)
    pool.map(find_non_overlapping_windows_pre_chr_no_graph, inputs)



def find_non_overlapping_windows_pre_chr_distributed(chrom, strand, this_ch_stra, save_path, threads):
    this_save_path = os.path.join(save_path, chrom+strand)
    # if chrom in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10'] and strand == '+':
    #     n_graphs = 2000
    # else:
    #     n_graphs = 1000

    n_graphs = 1000
    if not os.path.exists(this_save_path):
        os.mkdir(this_save_path)
    this_ch_stra = this_ch_stra.sort_values(by='win_start').reset_index(drop=True)

    this_ch_stra['interval'] = this_ch_stra.apply(lambda row: get_interval(row), axis=1)
    interval_list = list(this_ch_stra['interval'])

    # motif_intervals_list = list(zip(this_ch_stra.start, this_ch_stra.end))
    # windows_intervals_list = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    # interval_list = motif_intervals_list + windows_intervals_list
    # interval_list = list(set(interval_list))

    union = []
    for begin, end in sorted(interval_list):
        if union and union[-1][1] >= begin - 1:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([begin, end])

    union = sorted(union, key=lambda x: x[1])
    # select n_graphs out of all:
    splitted_unions = np.array_split(union, n_graphs)
    new_union = [(nu[0][0], nu[-1][-1]) for nu in splitted_unions]

    distributed_inputs = [[uni, chrom, strand, this_save_path, this_ch_stra] for uni in new_union]
    distributed_inputs = sorted(distributed_inputs, key=lambda x: x[0][0])
    print(len(distributed_inputs))

    for dist in distributed_inputs:
        compute_conflicting_nodes_in_interval(dist)

    pool = Pool(threads)
    pool.map(compute_conflicting_nodes_in_interval, distributed_inputs)
    print('Distributed graphs done.')

    conflicting_nodes = merge_conflicting_nodes_in_interval(this_save_path)
    conflicting_nodes = sorted(conflicting_nodes)
    if len(conflicting_nodes) > 0:
        with open(os.path.join(this_save_path, 'all_conflicting_nodes.txt'), 'w') as output_file:
            for node in conflicting_nodes:
                output_file.write(str(node) + '\n')

    print('Merging graphs done.')

    this_ch_stra['motif_interval'] = list(zip(this_ch_stra.start, this_ch_stra.end))
    this_ch_stra['window_interval'] = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    # non_overlapping_windows = this_ch_stra[(~this_ch_stra['window_interval'].isin(conflicting_nodes)) &
    # (~this_ch_stra['motif_interval'].isin(conflicting_nodes))]
    this_ch_stra = this_ch_stra.astype({'interval': str})
    if 'motif_len' in this_ch_stra:
        this_ch_stra = this_ch_stra.astype({'motif_len': int})

    non_overlapping_windows = this_ch_stra[~this_ch_stra['interval'].isin(conflicting_nodes)].reset_index(drop=True)
    non_overlapping_windows = non_overlapping_windows.sort_values(by='win_start').reset_index(drop=True)
    non_overlapping_windows.to_csv(os.path.join(this_save_path, 'non_overlapping_windows_' + chrom + strand + '.csv'), index=False)
    # conflicting_windows = this_ch_stra[(this_ch_stra['motif_interval'].isin(conflicting_nodes)) |
    # (this_ch_stra['window_interval'].isin(conflicting_nodes))]
    conflicting_windows = this_ch_stra[this_ch_stra['interval'].isin(conflicting_nodes)]
    conflicting_windows = conflicting_windows.sort_values(by='win_start').reset_index(drop=True)
    conflicting_windows.to_csv(os.path.join(this_save_path, 'conflicting_windows_' + chrom + strand + '.csv'), index=False)
    print('chromosome', chrom, 'strand', strand, 'done.')


def compute_conflicting_nodes_in_interval(inp):
    uni, chrom, strand, this_save_path, this_ch_stra = inp
    st = uni[0]
    en = uni[1]
    this_chunk = this_ch_stra[(this_ch_stra['win_end'] <= en + 1) & (this_ch_stra['end'] <= en + 1)].reset_index(drop=True)
    this_chunk = this_chunk[(this_chunk['win_start'] >= st - 1) & (this_chunk['start'] >= st - 1)].reset_index(drop=True)
    # print(uni, len(this_chunk))
    # motif_intervals_list_chunk = list(zip(this_chunk.start, this_chunk.end))
    # windows_intervals_list_chunk = list(zip(this_chunk.win_start, this_chunk.win_end))
    # interval_chunk = motif_intervals_list_chunk + windows_intervals_list_chunk
    interval_chunk = list(this_chunk['interval'])
    this_graph = nx.interval_graph(interval_chunk)
    # nx.write_gpickle(this_graph, os.path.join(this_save_path, 'interval_graph_' + chrom + strand + '_' + str(st)
    # + '_' + str(en) + '.gpickle'))
    # nx.write_gexf(this_graph, os.path.join(this_save_path, 'interval_graph_' + chrom + strand + '.gexf'))
    # G.number_of_edges()
    this_edges = this_graph.edges()
    this_conflicting_nodes = [item for sublist in list(this_edges) for item in sublist]
    this_conflicting_nodes = sorted(list(set(this_conflicting_nodes)))
    if len(this_conflicting_nodes) > 0:
        with open(os.path.join(this_save_path, chrom + strand + '_' + str(st) + '_' + str(en) + '.txt'), 'w') \
                as output_file:
            for node in this_conflicting_nodes:
                output_file.write(str(node) + '\n')


def merge_conflicting_nodes_in_interval(this_save_path):
    conflicting_nodes = []
    conf_files = [name for name in os.listdir(this_save_path) if '.txt' in name]
    for f in conf_files:
        with open(os.path.join(this_save_path, f), 'r') as fh:
            this_nodes = [line.split('\n')[0] for line in fh]
        conflicting_nodes += this_nodes
    for ff in conf_files:
        os.remove(os.path.join(this_save_path, ff))

    return conflicting_nodes


def find_non_overlapping_windows_pre_chr_no_graph(inp):
    chrom, strand, this_ch_stra, save_path = inp
    this_ch_stra['interval_start'] = this_ch_stra.apply(lambda row: get_interval(row)[0], axis=1)
    this_ch_stra['interval_end'] = this_ch_stra.apply(lambda row: get_interval(row)[1], axis=1)
    this_ch_stra = this_ch_stra.sort_values(by='interval_start').reset_index(drop=True)
    flag = 0
    this_ch_stra_non_overlapping = pd.DataFrame(columns=list(this_ch_stra.columns.values), index=range(len(this_ch_stra)))
    for i in range(len(this_ch_stra) - 1):
        if this_ch_stra.loc[i, 'interval_start'] > flag and \
                this_ch_stra.loc[i, 'interval_end'] < this_ch_stra.loc[i + 1, 'interval_start']:
            this_ch_stra_non_overlapping.loc[i, :] = this_ch_stra.loc[i, :]
        flag = max(flag, this_ch_stra.loc[i, 'interval_end'])
    this_ch_stra_non_overlapping = this_ch_stra_non_overlapping.dropna().reset_index(drop=True)
    this_ch_stra_non_overlapping.to_csv(os.path.join(save_path, 'non_overlapping_windows_' + chrom + '.csv'), index=False)
    print('chromosome', chrom, 'strand', strand, 'done.')


if __name__ == '__main__':
    
    if '-c' in sys.argv:
        chromo = sys.argv[sys.argv.index('-c') + 1]
    else:
        chromo = ''
        # exit('Error: Select a chromosome with -c.')
        
    if '-s' in sys.argv:
        strand = sys.argv[sys.argv.index('-s') + 1]
    else:
        strand = ''
        # exit('Error: Select a strand with -s.')
    
    if '-t' in sys.argv:
        threads = int(sys.argv[sys.argv.index('-t') + 1])
    else:
        threads = 1
    # step 1: prepare gff files from the non B DNA DB:
    # gff2bed_all_chr()
    
    # step 2: fix windows around motifs:
    # make_windows_on_motifs()
    
    # step 3: make windows in the opposite direction
    # extend_windows_to_other_strand_direction()
    
    # step 4: find the start of the first window and the end of last window on all chromosomes:
    # make_query_bounds_per_chr()
    
    # step 5: filter the reads in the interval
    # filter_query_samtools()
    
    # create windows steps:
    # step 6: make control windows
    # make_motif_free_intervals()
    
    # step 7: find non-overlapping windows
    find_non_overlapping_windows()
