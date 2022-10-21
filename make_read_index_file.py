import os
import pandas as pd


def read_summary_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        cols = lines[0].strip().split('\t')
        seq_df = pd.DataFrame(columns=cols, index=range(len(lines)-1))
        for i in range(1, len(lines)):
            seq_df.loc[i-1, :] = lines[i].strip().split('\t')
    return seq_df


def index_read_ids():
    main_fold = '/labs/Aguiar/non_bdna/DeepMod/hx1_ab231_2/'
    index_df = pd.DataFrame(columns=['filename', 'read_id', 'run_id', 'channel', 'start_time', 'duration', 'num_events', 'fast5_path', 'fastq_path'])
    batch_counter = str(0)
    main_folders = [os.path.join(main_fold, name) for name in os.listdir(main_fold) if os.path.isdir(os.path.join(main_fold, name))]
    for mf in main_folders:
        top_folders = [name for name in os.listdir(mf) if os.path.isdir(os.path.join(mf, name))]
        for tf in top_folders:
            summary_path = os.path.join(mf, tf, 'sequencing_summary.txt')
            if os.path.exists(summary_path):
                sum_df = read_summary_txt(summary_path)
                passed_df = sum_df.loc[sum_df['passes_filtering'] == 'True'].reset_index(drop=True)
                passed_df = passed_df.drop(columns=['passes_filtering', 'template_start', 'num_events_template',
                                                    'template_duration', 'num_called_template',
                                                    'sequence_length_template', 'mean_qscore_template',
                                                    'strand_score_template', 'calibration_strand_genome_template',
                                                    'calibration_strand_identity_template',
                                                    'calibration_strand_accuracy_template',
                                                    'calibration_strand_speed_bps_template'])
                passed_df['fast5_path'] = os.path.join(mf, tf, 'workspace', 'pass', 'ipds')
                passed_df['fastq_path'] = os.path.join(mf, tf, 'workspace', 'pass', batch_counter)
                index_df = index_df.append(passed_df, ignore_index=True)
    index_df = index_df.reset_index(drop=True)
    index_df.to_csv('read_ids_index.csv')


def make_fast5_index():
    '''Paper'''
    source_fast5_path = '/labs/Aguiar/non_bdna/human/na12878/fast5_preprocessed/'
    path = '/labs/Aguiar/non_bdna/human/na12878/tombo_alignments'
    mapping_file_path = '/labs/Aguiar/non_bdna/annotations/mapping_chr_chromosome.csv'
    mapping_file = pd.read_csv(mapping_file_path, index_col=False)
    chromosomes = list(mapping_file['chromosome'])
    chr_dict = {mapping_file.loc[i, 'chromosome']: mapping_file.loc[i, 'chr'] for i in range(len(mapping_file))}
    all_fast5_df = pd.DataFrame(columns=['mapped_chrom', 'mapped_strand', 'read_id', 'path'])
    files = sorted([name for name in os.listdir(path) if '.csv' in name])
    chunks = [files[x:x+300] for x in range(0, len(files), 300)]
    counter = 0
    for ch in chunks:
        for f in ch:
            df = pd.read_csv(os.path.join(path, f), index_col=0)
            # print(len(df), np.sum(df.duplicated()))
            # df = df[df['mapped_chrom'] != '0'].reset_index(drop=True)
            df = df[df['mapped_chrom'].isin(chromosomes)].reset_index(drop=True)
            counter += len(df)
            df = df.drop(columns=['mapped_start', 'mapped_end'])
            df['mapped_chrom'] = df['mapped_chrom'].apply(lambda x: chr_dict[x])
            df['read_id'] = df['filename'].apply(lambda x: x.split('.fast5')[0])
            folder_name = f.split('.csv')[0]
            fast5_path = os.path.join(source_fast5_path, folder_name)
            df['path'] = df['filename'].apply(lambda x: fast5_path + '/' + x)
            df = df.drop(columns=['filename'])
            all_fast5_df = all_fast5_df.append(df, ignore_index=True)
            print(folder_name, counter)
        all_fast5_df.to_csv('/labs/Aguiar/non_bdna/human/na12878/indices/fast5_index_' + str(counter) + '.csv')

    indices = [name for name in os.listdir('/labs/Aguiar/non_bdna/human/na12878/indices/') if '.csv' in name
               and not 'all' in name]
    all_df = pd.DataFrame(columns=['mapped_chrom', 'mapped_strand', 'read_id', 'path'])
    for ind in indices:
        this_df = pd.read_csv(os.path.join('/labs/Aguiar/non_bdna/human/na12878/indices/', ind), index_col=0)
        print(ind, len(this_df))
        all_df = all_df.append(this_df, ignore_index=True)
    all_df.to_csv('/labs/Aguiar/non_bdna/human/na12878/indices/all_fast5_index.csv')
    
    all_df = pd.read_csv('/labs/Aguiar/non_bdna/human/na12878/indices/all_fast5_index.csv', index_col=0)
    print('len of completer df', len(all_df))
    print(all_df[all_df.duplicated()])
    all_df_g = all_df.groupby(['mapped_chrom', 'mapped_strand']).count()
    groups = list(all_df_g.index)
    counter = 0
    for gr in groups:
        chr = gr[0]
        strand = gr[1]
        this_chr_strand = all_df[(all_df['mapped_chrom'] == chr) & (all_df['mapped_strand'] == strand)].reset_index(drop=True)
        this_chr_strand = this_chr_strand.drop(columns=['mapped_chrom', 'mapped_strand'])
        this_chr_strand = this_chr_strand.sort_values(by=['read_id']).reset_index(drop=True)
        this_chr_strand['path'] = this_chr_strand['path'].apply(lambda x: '/'.join(x.split('/')[:-1]) + '/workspace/pass/0/' + x.split('/')[-1])
        this_chr_strand = this_chr_strand.set_index(['read_id'], verify_integrity=True)
        this_chr_strand.to_csv('/labs/Aguiar/non_bdna/human/na12878/indices/' + chr + strand + '.csv')
        # counter += len(this_chr_strand)
        # print('length of this df', len(this_chr_strand), 'length of uniques', len(this_chr_strand['read_id'].unique()))
        # print('length of duplicates', np.sum(this_chr_strand.duplicated()))
        # print(this_chr_strand[this_chr_strand.duplicated()])
        # print(len(this_chr_strand) + counter)
        # counter += len(this_chr_strand)
        # this_chr_strand = this_chr_strand.set_index(['read_id'], verify_integrity=True)
        # this_chr_strand.to_csv('/labs/Aguiar/non_bdna/human/na12878/indices/' + chr + strand + '.csv', index=False)


def make_fast5_index_with_alignmet_info():
    '''Paper'''
    source_fast5_path = '/labs/Aguiar/non_bdna/human/na12878/fast5_preprocessed/'
    path = '/labs/Aguiar/non_bdna/human/na12878/tombo_alignments'
    mapping_file_path = '/labs/Aguiar/non_bdna/annotations/mapping_chr_chromosome.csv'
    mapping_file = pd.read_csv(mapping_file_path, index_col=False)
    chromosomes = list(mapping_file['chromosome'])
    chr_dict = {mapping_file.loc[i, 'chromosome']: mapping_file.loc[i, 'chr'] for i in range(len(mapping_file))}
    # all_fast5_df = pd.DataFrame(columns=['mapped_chrom', 'mapped_strand', 'mapped_start', 'mapped_end', 'read_id', 'path'])
    files = sorted([name for name in os.listdir(path) if '.csv' in name])
    chunks = [files[x:x+300] for x in range(0, len(files), 300)]
    counter = 0
    for ch in chunks:
        all_fast5_df = pd.DataFrame(columns=['mapped_chrom', 'mapped_strand', 'mapped_start', 'mapped_end', 'read_id', 'path'])
        for f in ch:
            df = pd.read_csv(os.path.join(path, f), index_col=0)
            # print(len(df), np.sum(df.duplicated()))
            # df = df[df['mapped_chrom'] != '0'].reset_index(drop=True)
            df = df[df['mapped_chrom'].isin(chromosomes)].reset_index(drop=True)
            # df = df.drop(columns=['mapped_start', 'mapped_end'])
            df['mapped_chrom'] = df['mapped_chrom'].apply(lambda x: chr_dict[x])
            df['read_id'] = df['filename'].apply(lambda x: x.split('.fast5')[0])
            folder_name = f.split('.csv')[0]
            fast5_path = os.path.join(source_fast5_path, folder_name, 'workspace', 'pass', '0')
            df['path'] = df['filename'].apply(lambda x: fast5_path + '/' + x)
            df = df.drop(columns=['filename'])
            all_fast5_df = all_fast5_df.append(df, ignore_index=True)
            print(folder_name, counter)
            
        all_fast5_df.to_csv('/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr/fast5_index_with_alignments_' + str(counter) + '.csv')
        counter += 1
        
    indices = [name for name in os.listdir('/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr/') if '.csv' in name
               and not 'all' in name]
    all_df = pd.DataFrame(columns=['mapped_chrom', 'mapped_strand', 'mapped_start', 'mapped_end', 'read_id', 'path'])
    for ind in indices:
        this_df = pd.read_csv(os.path.join('/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr/', ind), index_col=0)
        print(ind, len(this_df))
        all_df = all_df.append(this_df, ignore_index=True)
    all_df.to_csv('/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr/all_fast5_index_with_alignments.csv')
    
    all_df = pd.read_csv('/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr/all_fast5_index_with_alignments.csv', index_col=0)
    print('len of complete df', len(all_df))
    print(all_df[all_df.duplicated()])
    all_df_g = all_df.groupby(['mapped_chrom', 'mapped_strand']).count()
    groups = list(all_df_g.index)
    for gr in groups:
        chr = gr[0]
        strand = gr[1]
        this_chr_strand = all_df[(all_df['mapped_chrom'] == chr) & (all_df['mapped_strand'] == strand)].reset_index(drop=True)
        this_chr_strand = this_chr_strand.drop(columns=['mapped_chrom', 'mapped_strand'])
        this_chr_strand.to_csv('/labs/Aguiar/non_bdna/human/na12878/tombo_alignments_chr/' + chr + strand + '.csv')




if __name__ == '__main__':
    make_fast5_index()
