SLICES=${SLICES:-"0"} # SLICES could be set to one option "0 256 512 768", each option is a slice of randomly shuffled training data (of size 256)

METHOD=${1:-"SEED"}


seed_baseline_exp()
{
    NUM_TR=256
    NUM_TEST=1024
    SLICE_TR=$1

    if [ ! -z "$SLICE_TR" -a ! -z "$SLICE_TR" ]
    then
        OPENAI_API_KEY=${KEY} python run_selective.py --task strategyqa --num_dev ${NUM_TEST} --run_pred --slice_train ${SLICE_TR} --num_train ${NUM_TR} --batch_size 20 --style_template stdqa --do_fixedrand_scoring ${FLAG}
    fi
}



if [ "$METHOD" = "SEED" ]; then
    for SLI in ${SLICES}
    do
        seed_baseline_exp ${SLI}
    done
elif [ "$METHOD" = "OSACC" ]; then
    SLICES=${SLICES} sh exp_scripts/run_search/strategyqa.sh avgsilver
elif [ "$METHOD" = "OSLL" ]; then
    SLICES=${SLICES} sh exp_scripts/run_search/strategyqa.sh coherence
fi

