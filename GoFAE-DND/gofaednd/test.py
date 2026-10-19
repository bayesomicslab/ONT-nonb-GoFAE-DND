import os
import pandas as pd


main_path = '/labs/Aguiar/non_bdna/annotations/vae_windows/prepared_windows_req5/final_dataset_splitted/'
nonb_types = ['A_Phased_Repeat', 'Direct_Repeat','G_Quadruplex_Motif', 'Inverted_Repeat', 'Mirror_Repeat', 'Short_Tandem_Repeat', 'Z_DNA_Motif']
data_types = ['train', 'validation', 'test']
for nbt in nonb_types:
    for dt in data_types:
        data_path = os.path.join(main_path, nbt + '_' + dt + '.csv')
        df = pd.read_csv(data_path, index_col=0)
        n_total = len(df)
        n_bdna = len(df[df['label'] == 'bdna'])
        n_nbda = len(df[df['label'] == nbt])
        ratio = n_nbda/n_bdna
        print(nbt, dt)
        print('n_bdna:', n_bdna,'n_nonb:', n_nbda,'ratio:',ratio)


deltas = [20, 25, 30]
omegas = [0.5, 0.6, 0.7]
n_z_s =[32, 64]
for d in deltas:
    for w in omegas:
        for n_z in n_z_s:
            print(f'delta: {d}, omega: {w}, n_z: {n_z}')
            cmd = f'python Main.py --fQuiet --data_type simulated --nonb_type G_Quadruplex_Motif --nonb_ratio 0.1 --test sw --new_stat min --use_proj True --n_z {n_z} --latent_dim_proj 1 --num_projections 64 --batch_size 128 --outlier_batch_size 128 --alpha 0.3 --epochs 1 --use_labels True --poison_train False --poison_val True --use_rec_MH False --s1_pow 0.5 --s2_pow 0.5 --fdr_level 0.25 --unblock_test True --use_test_rob_mean_cov True --inform_plots True --loss_fn discriminative --discriminative_loss_fn mae --loss_fn_eval mae --discriminative_weight {d} --lambda_alpha 0.5 --omega {w} --Stiefel True --lr_e_d 1e-3 --beta1_enc_p1 0.9 --beta2_enc_p1 0.999 --beta1_dec 0.9 --beta2_dec 0.999 --h_dim 32 --momentum 0.01 --ncv 1 --num_workers 0'
            print(cmd)
            os.system(cmd)


