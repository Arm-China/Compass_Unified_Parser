#!/bin/bash

# check OPT if be cloned
if [ ! -d "../../Compass_OpportunePostTrainingTools" ]; then
  echo "Please clone Compass_OpportunePostTrainingTools project first to run example.."
  echo "You can find it here: https://github.com/Arm-China/Compass_OpportunePostTrainingTools"
  exit 255
fi

# suppose Compass_OpportunePostTrainingTools has been added in ../../
# add soft link to opt code
ln -s ../../Compass_OpportunePostTrainingTools/AIPUBuilder/Optimizer ../AIPUBuilder/