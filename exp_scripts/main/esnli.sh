SLICES=${SLICES:-"0"} # SLICES could be set to one option "0 1024 2048 3072", each option is a slice of randomly shuffled training data (of size 1024)

METHOD=${1:-"SEED"}

seed_baseline_exp()
{
    NUM_TR=1024
    NUM_TEST=1024
    NUM_SHOTS=9
    SLICE_TR=$1
    if [ ! -z "$SLICE_TR" -a ! -z "$SLICE_TR" ]
    then
        OPENAI_API_KEY=${KEY} python run_manual.py --task esnli --num_dev ${NUM_TEST} --run_pred --manual_prompt_id "clsbal_ntr${NUM_TR}_str${SLICE_TR}_nums${NUM_SHOTS}_stypsource_seed0" --batch_size 20 --style_template psource ${FLAG}
    fi
}


if [ "$METHOD" = "SEED" ]; then
    for SLI in ${SLICES}
    do
        seed_baseline_exp ${SLI}
    done
elif [ "$METHOD" = "OSACC" ]; then
    SLICES=${SLICES} sh exp_scripts/run_search/esnli.sh avgsilver
elif [ "$METHOD" = "OSLL" ]; then
    SLICES=${SLICES} sh exp_scripts/run_search/esnli.sh coherence
fi

