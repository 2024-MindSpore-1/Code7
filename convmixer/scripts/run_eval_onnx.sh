#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
SCRIPT_DIR=$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)
echo "SCRIPT_DIR=$SCRIPT_DIR"

display_usage() {
    echo -e "Usage: $0 DATA [--onnx_path ONNX_PATH]"
}

# Check if help in CLI arguments
for arg in "$@"
do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    display_usage
    exit 0
  fi
done

# Check if there are enough arguments
# If yes, parse the first three
if [[ $# -lt 1 ]]; then
    echo "Not enough arguments"
    exit 1
else
    DATA="$1"
    shift 1
fi

# Parse remain arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --onnx_path)
            if [[ "$#" -lt 2 ]]; then
                echo "No ONNX_PATH option"
                exit 1
            else
                ONNX_PATH="$2";
                shift
            fi ;;
        *) echo "Unknown option: '$1'"; exit 1 ;;
    esac
    shift
done

EVAL_SCRIPT_DIR="$SCRIPT_DIR/.."

# If variable CHECKPOINT is empty then evaluation can not be performed.
# Otherwise, run evaluation.
if [ -z "$ONNX_PATH" ]; then
    echo "Error! Expected --onnx_path option. "
    exit 1
fi

python3 "$EVAL_SCRIPT_DIR/eval_onnx.py" "$DATA" --onnx_path "$ONNX_PATH" 2>&1 | tee eval_onnx.log
