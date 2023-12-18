# GSM
NUM_TR=1024
NUM_DEV=1024

STRATEGY=${1:-random}
ARG_TEA_TIMES=${TEA_TIMES:-48}
ARG_STU_TIMES=${STU_TIMES:-16}
ARG_SPEED=${SPEED:-20}
ARG_SLICES=${SLICES:-"0 1024 2048 3072"}

if [ "$MODE" = "scan" ]; then
    ARG_FLAG="--do_inspect"
elif [ "$MODE" = "score" ]; then
    ARG_FLAG="--do_score_only"
fi

echo ${STRATEGY}

search_exp()
{
    SLICE_TR=$1
    if [ ! -z "$SLICE_TR" -a ! -z "$SLICE_TR" ]
    then
        OPENAI_API_KEY=${KEY}  python expl_search/run_unsup_search.py --task gsm --num_dev ${NUM_DEV} --eng code-davinci-002 --slice_train ${SLICE_TR} --num_train ${NUM_TR} --aug_num_samples 40 --aug_temperature 0.7 --run_pred --tune_split train --num_tune 256 --teacher_times_search ${ARG_TEA_TIMES} --teacher_temperature 0.7 --teacher_num_samples 1 --teacher_batch_size ${ARG_SPEED} --student_search_strategy ${STRATEGY} --student_times_search ${ARG_STU_TIMES} --student_num_samples 1 --student_temperature 0.0 --student_batch_size ${ARG_SPEED} --test_num_samples 1 --test_temperature 0.0 --batch_size 20 ${ARG_FLAG} ${FLAG}
    fi
}

unsup_search_table()
{
    for SLI in ${ARG_SLICES}
    do
        search_exp ${SLI}
    done
}

unsup_search_table
