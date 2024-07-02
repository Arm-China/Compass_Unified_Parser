SRC_DIR=/project/ai/zhouyi_compass/Model_Resource/huggingface/mistralai/Mistral-7B-v0.1
OUTPUT_DIR=/project/ai/zhouyi_compass/Model_Resource/huggingface/mistralai/Mistral-7B-v0.1/OUTPUTS/ONNX

optimum-cli export onnx --model $SRC_DIR --task text-generation $OUTPUT_DIR