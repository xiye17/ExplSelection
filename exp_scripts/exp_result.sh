DATASET=$1 # gsm ecqa esni strategyqa
SETS=${SETS:-"0"} # sets can be 0 1 2 3
NUM_TEST=1024

if [ "$DATASET" = "gsm" ]; then
    STYLE_TPL="default"
elif [ "$DATASET" = "ecqa" ]; then
    STYLE_TPL="default"
elif [ "$DATASET" = "esnli" ]; then
    STYLE_TPL="psource"
elif [ "$DATASET" = "strategyqa" ]; then
    STYLE_TPL="stdqa"
else
    echo "Dataset not supported"
    exit 1
fi

for SET_ID in ${SETS}
do
    echo "SEED"
    OPENAI_API_KEY=${KEY} python run_manual.py --task ${DATASET} --num_dev ${NUM_TEST} --run_pred --manual_prompt_id "${DATASET}_seed${SET_ID}" --batch_size 20 --style_template ${STYLE_TPL} ${FLAG}

    echo "SEACHED"
    OPENAI_API_KEY=${KEY} python run_manual.py --task ${DATASET} --num_dev ${NUM_TEST} --run_pred --manual_prompt_id "${DATASET}_searched${SET_ID}" --batch_size 20 --style_template ${STYLE_TPL} ${FLAG}

done
