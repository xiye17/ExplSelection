SLICES=${SLICES:-"0"} # SLICES could be set to one option "0 1024 2048 3072", each option is a slice of randomly shuffled training data (of size 1024)

METHOD=${1:-"SEED"}

seed_baseline_exp()
{
    NUM_TR=1024
    NUM_TEST=1024
    SLICE_TR=$1
    if [ ! -z "$SLICE_TR" -a ! -z "$SLICE_TR" ]
    then
        OPENAI_API_KEY=${KEY} python run_selective.py --task gsm --num_dev ${NUM_TEST} --run_pred --slice_train ${SLICE_TR} --num_train ${NUM_TR} --batch_size 20 --do_fixedrand_scoring ${FLAG}
    fi
}


if [ "$METHOD" = "SEED" ]; then
    for SLI in ${SLICES}
    do
        seed_baseline_exp ${SLI}
    done
elif [ "$METHOD" = "OSACC" ]; then
    SLICES=${SLICES} sh exp_scripts/run_search/gsm.sh avgsilver
elif [ "$METHOD" = "OSLL" ]; then
    SLICES=${SLICES} sh exp_scripts/run_search/gsm.sh coherence
fi

