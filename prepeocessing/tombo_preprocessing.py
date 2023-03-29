import tombo
import h5py
import numpy as np
from tombo import tombo_helper, tombo_stats, resquiggle
import pickle

def convert_to_pA_numpy(d, digitisation, fast5_range, offset):
    raw_unit = fast5_range / digitisation
    return (d + offset) * raw_unit

reg_data = tombo_helper.intervalData(chrm='NC_000019.10', start=4209765, end=4210109, strand='-')

fast5_path = 'ffb4be6f-8aa9-43b5-987c-f7c87d8fad9f.fast5'

reads_index = tombo_helper.TomboReads(['/mnt/d/UCONN/nonBDNA',])




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




tombo_events = list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'])
tombo_moves = [ev[3] for ev in tombo_events]
tombo_seq = ''.join([ev[4].decode() for ev in tombo_events])
num_del = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_deletions')
num_ins = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_insertions')
num_match = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_matches')
num_mismatch = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('num_mismatches')
mapped_start = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_start')
mapped_end = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_end')
clipped_bases_start = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('clipped_bases_start')
clipped_bases_end = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('clipped_bases_end')
mapped_chrom = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_chrom')
mapped_strand = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Alignment'].attrs.get('mapped_strand')
basecall_group = f['Analyses/RawGenomeCorrected_000'].attrs.get('basecall_group')
tombo_version = f['Analyses/RawGenomeCorrected_000'].attrs.get('tombo_version')
scale = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('scale')
shift = f['Analyses/RawGenomeCorrected_000/BaseCalled_template'].attrs.get('shift')
rel_idx = f['Analyses/RawGenomeCorrected_000/BaseCalled_template/Events'].attrs.get('read_start_rel_to_raw')

list(f['Analyses/RawGenomeCorrected_000/BaseCalled_template'])
list(f['Analyses/RawGenomeCorrected_000'].attrs.keys())
norm_raw = (raw - shift)/scale


index_file_path = '.0.RawGenomeCorrected_000.tombo.index'
with open(index_file_path, 'rb') as f:
    index_file = pickle.load(f)

keys = list(index_file.keys())

read_count = 0
for key in keys:
    read_count += len(index_file[key])

for i in range(len(index_file[(mapped_chrom, mapped_strand)])):
    if index_file[(mapped_chrom, mapped_strand)][i][0] == fast5_path:
        print(index_file[(mapped_chrom, mapped_strand)][i])

from scipy.stats import norm
import random
noise = random.normal(loc=0.0, scale=0.1, size=100)
x = range(0,100)
rv = norm()
rv.pdf(x)
aa = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),
        'r-', lw=5, alpha=0.6, label='norm pdf')
