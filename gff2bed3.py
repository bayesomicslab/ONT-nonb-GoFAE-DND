import sys
import os
import networkx as nx
import pandas as pd
import numpy as np
from multiprocessing import Pool

def gff2bed2(main_folder):
    gff_files = [name for name in os.listdir(main_folder) if '.gff' in name]
    for gf in gff_files:
        gff_df = pd.read_csv(os.path.join(main_folder, gf), names=['chr', 'source', 'feature', 'start', 'end', 'score',
                                                                   'strand', 'frame', 'attribute'], skiprows=[0], sep='\t')
        # gff_df = read_gff(os.path.join(main_folder, gf))
        gff_df = gff_df.drop(columns=['source', 'feature', 'score', 'strand', 'frame', 'attribute'])
        gff_df = gff_df.dropna()
        name = gf.split('.gff')[0]+'.bed'
        gff_df.to_csv(os.path.join(main_folder, name), sep='\t', index=False, header=False)

def gff2csv(main_folder):
    gff_files = [name for name in os.listdir(main_folder) if '.gff' in name]
    for gf in gff_files:
        gff_df = pd.read_csv(os.path.join(main_folder, gf), names=['chr', 'source', 'feature', 'start', 'end', 'score',
                                                                   'strand', 'frame', 'attribute'], skiprows=[0], sep='\t')
        # gff_df = read_gff(os.path.join(main_folder, gf))
        gff_df = gff_df.drop(columns=['source', 'score', 'frame'])
        gff_df = gff_df.dropna()
        name = gf.split('.gff')[0]+'.csv'
        gff_df.to_csv(os.path.join(main_folder, name), sep=',', index=False, header=False)


def gff_chromosomes(main_folder, chrom):
    gff_files = [name for name in os.listdir(main_folder) if '.gff' in name and chrom in name]
    chromosome_df = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand', 'attribute'])
    for gf in gff_files:
        gff_df = pd.read_csv(os.path.join(main_folder, gf), names=['chr', 'source', 'feature', 'start', 'end', 'score',
                                                                   'strand', 'frame', 'attribute'], skiprows=[0], sep='\t')
        gff_df = gff_df.drop(columns=['source', 'score', 'frame'])
        gff_df = gff_df.dropna().reset_index(drop=True)
        chromosome_df = chromosome_df.append(gff_df, ignore_index=True)
    name = chrom + '.csv'
    chromosome_df.to_csv(os.path.join(main_folder, name), sep=',', index=False, header=False)

def coverage_chromosome(main_folder, chrom):
    txt_files = [name for name in os.listdir(main_folder) if '.txt' in name and chrom in name]
    chromosome_df = pd.DataFrame(columns=['chr', 'position', 'coverage'])
    for tf in txt_files:
        tf_df = pd.read_csv(os.path.join(main_folder, tf), names=['chr', 'position', 'coverage'], sep='\t')
        chromosome_df = chromosome_df.append(tf_df, ignore_index=True)
    name = chrom + '.txt'
    chromosome_df.to_csv(os.path.join(main_folder, name), sep=',', index=False, header=False)


def intersect(int1, int2):
    return (max(int1[0], int2[0]), min(int1[1], int2[1])) > 0


def get_interval_graph():
    path = 'Data\chr22.csv'
    chromosome_df = pd.read_csv(path, names=['chr', 'feature', 'start', 'end', 'strand', 'attribute'])
    # chromosome_df = chromosome_df.drop(columns=['chr', 'feature', 'strand', 'attribute'])
    chromosome_df = chromosome_df.sort_values(by='start').reset_index(drop=True)
    chromosome_df['interval'] = list(zip(chromosome_df.start, chromosome_df.end))
    intervals = list(chromosome_df['interval'])

    G = nx.interval_graph(intervals)
    G.number_of_edges()
    return G


def fix_windows_on_motifs(ms, me):
    """ms: motif start, me: motif end"""
    ml = me - ms + 1
    if ml%2 == 0: # motif length is even
        a = ml/2
        b = 50 - a
        win_s = ms-b
        win_e = me+b
    else: # ml%2 == 1: # motif length is odd
        a = np.floor(ml/2)
        b_start = 50 - (a + 1)
        b_end = 50 - a
        win_s = ms - b_start
        win_e = me + b_end
    return [win_s, win_e]


def gff2bed2_all_chr(main_folder):
    main_folder = '/labs/Aguiar/non_bdna/annotations/human_hg38.gff'
    gff_files = [name for name in os.listdir(main_folder) if '.gff' in name]
    add_non_b_ann = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand'])
    for gf in gff_files:
        gff_df = pd.read_csv(os.path.join(main_folder, gf), names=['chr', 'source', 'feature', 'start', 'end', 'score',
                                                                   'strand', 'frame', 'attribute'], skiprows=[0], sep='\t')
        # gff_df = read_gff(os.path.join(main_folder, gf))
        gff_df = gff_df.drop(columns=['source', 'score', 'frame', 'attribute'])
        gff_df = gff_df.dropna()
        add_non_b_ann = add_non_b_ann.append(gff_df, ignore_index=True).reset_index(drop=True)
    add_non_b_ann.to_csv('/labs/Aguiar/non_bdna/annotations/all_non_b.csv', index=False)
    add_non_b_ann['motif_len'] = add_non_b_ann.apply(lambda row: row.end - row.start + 1, axis=1)
    add_non_b_ann['win_start'] = add_non_b_ann.apply(lambda row: fix_windows_on_motifs(row.start, row.end)[0], axis=1)
    add_non_b_ann['win_end'] = add_non_b_ann.apply(lambda row: fix_windows_on_motifs(row.start, row.end)[1], axis=1)
    convert_dict = {'win_start': int, 'win_end': int}
    add_non_b_ann = add_non_b_ann.astype(convert_dict)
    add_non_b_ann.to_csv('/labs/Aguiar/non_bdna/annotations/all_non_b_windows.csv', index=False)
    
    non_b_g_min = add_non_b_ann.groupby(['chr', 'strand'])['win_start'].min()
    non_b_g_max = add_non_b_ann.groupby(['chr', 'strand'])['win_end'].max()
    groups = list(non_b_g_min.index)
    queries_df = pd.DataFrame(columns=['chr', 'strand', 'start', 'end'], index=range(len(groups)))
    i = 0
    for gr in groups:
        chr = gr[0]
        strand = gr[1]
        start = non_b_g_min.loc[gr]
        end = non_b_g_max.loc[gr]
        queries_df.loc[i, :] = chr, strand, start, end
        i += 1
    queries_df.to_csv('/labs/Aguiar/non_bdna/annotations/chr_start_end_queries.csv', index=False)


def make_motif_free_intervals():
    path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows.csv'
    # path = 'all_non_b_windows.csv'
    windows = pd.read_csv(path)
    # save_path = '/labs/Aguiar/non_bdna/annotations/windows/control_conservative/'
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/control/'
    
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
    pool.map(make_control_windows_chr_strand, chr_strand_windows_inputs)
    # pool.map(make_control_windows_chr_strand_conservative, chr_strand_windows_inputs)
    
    control_windows = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand', 'motif_len', 'win_start',
                                            'win_end'])
    control_files = [name for name in os.listdir(save_path) if '.csv' in name]
    for cf in control_files:
        cf_df = pd.read_csv(os.path.join(save_path, cf), index_col=0)
        control_windows = control_windows.append(cf_df, ignore_index=True)
    # control_windows.to_csv(os.path.join(save_path, 'all_controls_conservative.csv'))
    control_windows.to_csv(os.path.join(save_path, 'all_controls.csv'))


def make_control_wins_per_interval(chrom, stra, this_free):
    # not conservative
    motif_free_df = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand', 'motif_len', 'win_start',
                                          'win_end'], index=range(100000))
    st = this_free[0] + 20
    en = this_free[1] - 20
    # wins = []
    sta = st
    counter = 0
    while sta + 100 < en:
        # interv = (sta, sta + 99)
        # wins.append(interv)
        motif_free_df.loc[counter, :] = chrom, 'control', sta, sta + 99, stra, 100, sta, sta + 99
        counter += 1
        sta += 100
    motif_free_df = motif_free_df.dropna()
    # motif_free_df.to_csv('Data/' +chr + stra + '_control.csv', header=None, index=None)
    
    return motif_free_df


def make_control_windows_chr_strand(inp):
    # not conservative
    chr, stra, this_ch_stra, save_path = inp
    motif_intervals_list = list(zip(this_ch_stra.start, this_ch_stra.end))
    windows_intervals_list = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    interval_list = motif_intervals_list + windows_intervals_list
    
    # intervals_list = [(7, 9), (11, 13), (11, 15), (14, 20), (23, 39)]
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
        free = [union[i][1] + 1, union[i+1][0]-1]
        if free[1] - free[0] > 140:
            # print(free[1] - free[0])
            motif_free_intervals.append(free)

    # make the windows (bed file)
    control_df = pd.DataFrame(columns=['chr', 'feature', 'start', 'end', 'strand', 'motif_len', 'win_start', 'win_end'])
    for i in range(len(motif_free_intervals)):
        this_free = motif_free_intervals[i]
        this_df = make_control_wins_per_interval(chr, stra, this_free)
        control_df = control_df.append(this_df, ignore_index=True)
    control_df.to_csv(os.path.join(save_path, 'control_' + chr + stra + '.csv'))


def make_control_windows_chr_strand_conservative(inp):
    # conservative
    chr, stra, this_ch_stra, save_path = inp
    
    motif_intervals_list = list(zip(this_ch_stra.start, this_ch_stra.end))
    windows_intervals_list = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    interval_list = motif_intervals_list + windows_intervals_list
    
    # intervals_list = [(7, 9), (11, 13), (11, 15), (14, 20), (23, 39)]
    # find unions of regions
    
    union = []
    for begin,end in sorted(interval_list):
        if union and union[-1][1] >= begin - 1:
            union[-1][1] = max(union[-1][1], end)
        else:
            union.append([begin, end])
    
    union = sorted(union, key=lambda x: x[1])
    # find between regions
    motif_free_intervals = []
    for i in range(len(union) - 1):
        free = [union[i][1] + 1, union[i+1][0]-1]
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
        win_start = mid-49
        win_end = mid+50
        # tt.append((win_start, win_end))
        # print(motif_free_intervals[i],'==>',(win_start, win_end))
        control_df.loc[i, :] = chr, 'Control', win_start, win_end, stra, 100, win_start, win_end
    
    control_df.to_csv(os.path.join(save_path, 'control_conservative_' + chr + stra + '.csv'))


def find_non_overlapping_windows():
    path = '/labs/Aguiar/non_bdna/annotations/all_non_b_windows.csv'
    # path = 'all_non_b_windows.csv'
    
    control_path = '/labs/Aguiar/non_bdna/annotations/windows/control_conservative/all_controls_conservative.csv'
    # control_path = 'all_controls_conservative.csv'
    
    # save_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping'
    save_path = '/labs/Aguiar/non_bdna/annotations/windows/non_overlapping2'
    
    windows = pd.read_csv(path)
    controls = pd.read_csv(control_path, index_col=0)
    # controls.groupby(['chr', 'strand']).count().sort_values(['chr', 'strand'])
    
    all_windows = windows.append(controls, ignore_index=True).reset_index(drop=True)
    chr_strand_windows_inputs = []
    non_b_groups = all_windows.groupby(['chr', 'strand']).count()
    groups = list(non_b_groups.index)
    for gr in groups:
        chrom, stra = gr
        # print(chr, stra)
        this_ch_stra = all_windows[(all_windows['chr'] == chrom) & (all_windows['strand'] == stra)].reset_index(drop=True)
        print(chrom, stra, len(this_ch_stra))
        chr_strand_windows_inputs.append([chrom, stra, this_ch_stra, save_path])
    
    pool = Pool(40)
    pool.map(find_non_overlapping_windows_pre_chr, chr_strand_windows_inputs)


def make_interval_graph_from_winodws(df):
    win_intervals = list(zip(df.win_start, df.win_end))
    motif_intervals = list(zip(df.start, df.end))
    interval = motif_intervals + win_intervals
    interval = list(set(interval))
    graph = nx.interval_graph(interval)
    return graph


def get_interval(row):
    if row.motif_len > 100:
        biger_interval = (row.start, row.end)
    else:
        biger_interval = (row.win_start, row.win_end)
    return biger_interval


def find_non_overlapping_windows_pre_chr(inp):
    chrom, stra, this_ch_stra, save_path = inp
    motif_intervals_list = list(zip(this_ch_stra.start, this_ch_stra.end))
    windows_intervals_list = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    interval_list = motif_intervals_list + windows_intervals_list
    interval_list = list(set(interval_list))
    # inter_lst =[(1, 5), (6, 8), (9, 14), (14, 18), (1, 5), (4,7)]
    graph = nx.interval_graph(interval_list)
    nx.write_gpickle(graph, os.path.join(save_path, 'interval_graph_' + chrom + stra + '.gpickle'))
    nx.write_gexf(graph, os.path.join(save_path, 'interval_graph_' + chrom + stra + '.gexf'))
    # G.number_of_edges()
    edges = graph.edges()
    conflicting_nodes = [item for sublist in list(edges) for item in sublist]
    conflicting_nodes = list(set(conflicting_nodes))

    conflicting_nodes = sorted(conflicting_nodes, key=lambda x: x[0])

    # this_ch_stra['motif_interval'] = list(zip(this_ch_stra.start, this_ch_stra.end))
    this_ch_stra.loc[:, 'motif_interval'] = list(zip(this_ch_stra.start, this_ch_stra.end))
    # this_ch_stra['window_interval'] = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    this_ch_stra.loc[:, 'window_interval'] = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    non_overlapping_windows = this_ch_stra[(~this_ch_stra['window_interval'].isin(conflicting_nodes)) & (~this_ch_stra['motif_interval'].isin(conflicting_nodes))]
    non_overlapping_windows.to_csv(os.path.join(save_path, 'non_overlapping_windows_' + chrom + stra + '.csv'), index=False)
    conflicting_windows = this_ch_stra[(this_ch_stra['motif_interval'].isin(conflicting_nodes)) | (this_ch_stra['window_interval'].isin(conflicting_nodes))]
    conflicting_windows.to_csv(os.path.join(save_path, 'conflicting_windows_' + chrom + stra + '.csv'), index=False)
    print('chromosome', chrom, 'strand', stra, 'done.')


def find_non_overlapping_windows_pre_chr_no_graph(inp):
    chrom, stra, this_ch_stra, save_path = inp
    motif_intervals_list = list(zip(this_ch_stra.start, this_ch_stra.end))
    windows_intervals_list = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    interval_list = motif_intervals_list + windows_intervals_list
    interval_list = list(set(interval_list))


    conflicting_nodes = [item for sublist in list(edges) for item in sublist]
    conflicting_nodes = list(set(conflicting_nodes))
    
    conflicting_nodes = sorted(conflicting_nodes, key=lambda x: x[0])
    
    # this_ch_stra['motif_interval'] = list(zip(this_ch_stra.start, this_ch_stra.end))
    this_ch_stra.loc[:, 'motif_interval'] = list(zip(this_ch_stra.start, this_ch_stra.end))
    # this_ch_stra['window_interval'] = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    this_ch_stra.loc[:, 'window_interval'] = list(zip(this_ch_stra.win_start, this_ch_stra.win_end))
    non_overlapping_windows = this_ch_stra[(~this_ch_stra['window_interval'].isin(conflicting_nodes)) & (~this_ch_stra['motif_interval'].isin(conflicting_nodes))]
    non_overlapping_windows.to_csv(os.path.join(save_path, 'non_overlapping_windows_' + chrom + stra + '.csv'), index=False)
    conflicting_windows = this_ch_stra[(this_ch_stra['motif_interval'].isin(conflicting_nodes)) | (this_ch_stra['window_interval'].isin(conflicting_nodes))]
    conflicting_windows.to_csv(os.path.join(save_path, 'conflicting_windows_' + chrom + stra + '.csv'), index=False)
    print('chromosome', chrom, 'strand', stra, 'done.')


if __name__ == '__main__':
    if '-a' in sys.argv:
        my_path = sys.argv[sys.argv.index('-a') + 1]
    else:
        my_path = ''
        # exit('Error: Select the main directory using  -a.')
    if '-c' in sys.argv:
        chromo = sys.argv[sys.argv.index('-c') + 1]
    else:
        chromo = 'chr2222'
    # gff_chromosomes(my_path, chr)
    make_motif_free_intervals()
    # find_non_overlapping_windows()
