#!/bin/bash
########## setup project directory ##########
CODE_DIR=`realpath $(dirname "$0")/..`
echo "Locate the project folder at ${CODE_DIR}"
cd ${CODE_DIR}

######### check number of args ##########
HELP="Usage example: GPU=0 bash $0 <name> <AE config> <PreLDM config> <LDM config> <latent dist config> <test config> [mode: e.g. 11111]"
if [ -z $1 ]; then
    echo "Experiment name missing. ${HELP}"
    exit 1;
else
    NAME=$1
fi

if [ -z $2 ]; then
    echo "Autoencoder config missing. ${HELP}"
    exit 1;
else
    AECONFIG=$2
fi

if [ -z $3 ]; then
    echo "PreLDM config missing. ${HELP}"
    exit 1;
else
    PRELDMCONFIG=$3
fi

if [ -z $4 ]; then
    echo "LDM config missing. ${HELP}"
    exit 1;
else
    LDMCONFIG=$4
fi

if [ -z $5 ]; then
    echo "Latent distance config missing. ${HELP}"
    exit 1;
else
    LATENT_DIST_CONFIG=$5
fi

if [ -z $6 ]; then
    echo "Test config missing. ${HELP}"
    exit 1;
else
    TEST_CONFIG=$6
fi

if [ -z $7 ]; then
    MODE=11111
else
    MODE=$7
fi

echo "Mode: $MODE, [train AE] / [pretrain LDM] / [train LDM] / [Generate] / [Evaluation]"

TRAIN_AE_FLAG=${MODE:0:1}
PRETRAIN_LDM_FLAG=${MODE:1:1}
TRAIN_LDM_FLAG=${MODE:2:1}
GENERATE_FLAG=${MODE:3:1}
EVAL_FLAG=${MODE:4:1}

########## setup directories ##########
AE_SAVE_DIR=./exp/$NAME/AE
PRE_LDM_SAVE_DIR=./exp/$NAME/PreLDM
LDM_SAVE_DIR=./exp/$NAME/LDM
OUTLOG=./exp/$NAME/output.log

if [[ ! -e ./exp/$NAME ]]; then
    mkdir -p ./exp/$NAME
elif [[ -e $AE_SAVE_DIR ]] && [ "$TRAIN_AE_FLAG" = "1" ]; then
    echo "Directory ${AE_SAVE_DIR} exists! But training flag is 1!"
    exit 1;
elif [[ -e $PRE_LDM_SAVE_DIR ]] && [ "$PRETRAIN_LDM_FLAG" = "1" ]; then
    echo "Directory ${PRE_LDM_SAVE_DIR} exists! But training flag is 1!"
    exit 1;
elif [[ -e $LDM_SAVE_DIR ]] && [ "$TRAIN_LDM_FLAG" = "1" ]; then
    echo "Directory ${LDM_SAVE_DIR} exists! But training flag is 1!"
    exit 1;
fi

GPU=${GPU:-0}

########## train autoencoder ##########
echo "Training Autoencoder with config $AECONFIG:" | tee $OUTLOG
cat $AECONFIG | tee -a $OUTLOG
if [ "$TRAIN_AE_FLAG" = "1" ]; then
    bash scripts/train.sh $AECONFIG --trainer.config.save_dir=$AE_SAVE_DIR
fi
echo "" | tee -a $OUTLOG

########## pretrain ldm ##########
echo "Pretraining LDM with config $PRELDMCONFIG:" | tee -a $OUTLOG
cat $PRELDMCONFIG | tee -a $OUTLOG
AE_CKPT=`cat ${AE_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using Autoencoder checkpoint: ${AE_CKPT}" | tee -a $OUTLOG

if [ "$PRETRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $PRELDMCONFIG --trainer.config.save_dir=$PRE_LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT
fi
echo "" | tee -a $OUTLOG

########## train ldm ##########
echo "Training LDM with config $LDMCONFIG:" | tee -a $OUTLOG
cat $LDMCONFIG | tee -a $OUTLOG
PRELDM_CKPT=`cat ${PRE_LDM_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Using pretrained LDM checkpoint: ${PRELDM_CKPT}" | tee -a $OUTLOG

if [ "$TRAIN_LDM_FLAG" = "1" ]; then
    bash scripts/train.sh $LDMCONFIG --trainer.config.save_dir=$LDM_SAVE_DIR --model.autoencoder_ckpt=$AE_CKPT --load_ckpt=$PRELDM_CKPT
fi
echo "" | tee -a $OUTLOG

########## get latent distance ##########
LDM_CKPT=`cat ${LDM_SAVE_DIR}/version_0/checkpoint/topk_map.txt | head -n 1 | awk -F " " '{print $2}'`
echo "Get distances in latent space" | tee -a $OUTLOG
python setup_latent_guidance.py --config ${LATENT_DIST_CONFIG} --ckpt ${LDM_CKPT} --gpu ${GPU:0:1} >> $OUTLOG
echo "" | tee -a $OUTLOG

########## generate ##########
echo "Generate results Using LDM checkpoint: ${LDM_CKPT}" | tee -a $OUTLOG
if [ "$GENERATE_FLAG" = "1" ]; then
    python generate.py --config $TEST_CONFIG --ckpt $LDM_CKPT --gpu ${GPU:0:1}
fi
echo "" | tee -a $OUTLOG

########## cal metrics ##########
if [ "$EVAL_FLAG" = "1" ]; then
    echo "Evaluation:" | tee -a $OUTLOG
    python cal_metrics.py --results ${LDM_SAVE_DIR}/version_0/results/results.jsonl >> $OUTLOG
fi
