#!/bin/bash



python3 train_eval_stca_late_elementwise_sum.py -e 71 -mk 4096 -mm 0.0 -ff 4096 -od 1024 -ffs 1024 -tfs 1024 -mp /mldisk/nfs_shared_/dh/weights/vcdb-byol_rmac-segment-1024+1024-late+plus-wiz -fp /workspace/CTCA/pre_processing/vcdb-byol_rmac_89325.hdf5 -sp /workspace/CTCA/pre_processing/vcdb-segment_l2norm_89325.hdf5 -efp /workspace/CTCA/pre_processing/fivr-byol_rmac_segment_l2norm_1024+1024.hdf5 -effp /workspace/CTCA/pre_processing/fivr-byol_rmac_187563.hdf5 -efsp /workspace/CTCA/pre_processing/fivr-segment_l2norm_187260_1024.hdf5;

