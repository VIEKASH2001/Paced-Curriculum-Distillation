# #!/usr/bin/env bash

# initial_mu=0.2780
# for mu_update in 0.222 0.074 0.044; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

# initial_mu=0.3165
# for mu_update in 0.184 0.061 0.037; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

# initial_mu=0.3765
# for mu_update in 0.124 0.041 0.025; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

#!/usr/bin/env bash

# initial_mu=0.39
# for mu_update in -0.183 -0.55 0.55 0.183; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

# initial_mu=0.42
# for mu_update in -0.133 -0.4 0.4 0.133; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

# initial_mu=0.44
# for mu_update in -0.1 -0.3 0.3 0.1; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

# initial_mu=0.46
# for mu_update in -0.067 -0.2 0.2 0.067; do
#     python train_model.py \
#         --epochs 40 \
#         --model-path "outputs/bus/ckd-spl/${initial_mu}_${mu_update}/unet.pth" \
#         --tb \
#         --spl \
#         --mu-update ${mu_update} \
#         --initial-mu ${initial_mu} \
#         --ckd-loss-type both_weighted_spl_per_px_no_alpha
# done

initial_mu=0.39
for mu_update in -0.183 -0.55 0.55 0.183; do
    python train_model.py \
        --epochs 40 \
        --model-path "outputs/bus/ckd-spl-no-weights/${initial_mu}_${mu_update}/unet.pth" \
        --tb \
        --spl \
        --mu-update ${mu_update} \
        --initial-mu ${initial_mu} \
        --ckd-loss-type both_weighted_spl_per_px_no_alpha_no_weights
done

initial_mu=0.42
for mu_update in -0.133 -0.4 0.4 0.133; do
    python train_model.py \
        --epochs 40 \
        --model-path "outputs/bus/ckd-spl-no-weights/${initial_mu}_${mu_update}/unet.pth" \
        --tb \
        --spl \
        --mu-update ${mu_update} \
        --initial-mu ${initial_mu} \
        --ckd-loss-type both_weighted_spl_per_px_no_alpha_no_weights
done

initial_mu=0.44
for mu_update in -0.1 -0.3 0.3 0.1; do
    python train_model.py \
        --epochs 40 \
        --model-path "outputs/bus/ckd-spl-no-weights/${initial_mu}_${mu_update}/unet.pth" \
        --tb \
        --spl \
        --mu-update ${mu_update} \
        --initial-mu ${initial_mu} \
        --ckd-loss-type both_weighted_spl_per_px_no_alpha_no_weights
done

initial_mu=0.46
for mu_update in -0.067 -0.2 0.2 0.067; do
    python train_model.py \
        --epochs 40 \
        --model-path "outputs/bus/ckd-spl-no-weights/${initial_mu}_${mu_update}/unet.pth" \
        --tb \
        --spl \
        --mu-update ${mu_update} \
        --initial-mu ${initial_mu} \
        --ckd-loss-type both_weighted_spl_per_px_no_alpha_no_weights
done