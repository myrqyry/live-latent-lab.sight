#!/bin/bash
# Script to download ONNX Runtime for Linux x64
# Version 1.17.1 (Matching the version found on system or a known stable one)

ORT_VERSION="1.17.1"
ORT_FILE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"

EXT_DIR="$(pwd)/external"

mkdir -p "$EXT_DIR"
cd "$EXT_DIR"

if [ -d "onnxruntime-linux-x64-${ORT_VERSION}" ]; then
    echo "ONNX Runtime ${ORT_VERSION} already exists in $EXT_DIR"
    exit 0
fi

echo "Downloading ONNX Runtime ${ORT_VERSION}..."
curl -L -O "$ORT_URL"

if [ -f "$ORT_FILE" ]; then
    echo "Extracting..."
    tar -xzf "$ORT_FILE"
    rm "$ORT_FILE"
    
    # Create a generic symlink for easier CMake usage
    ln -sf "onnxruntime-linux-x64-${ORT_VERSION}" "onnxruntime"
    
    echo "ONNX Runtime installed to $EXT_DIR/onnxruntime"
else
    echo "Download failed."
    exit 1
fi
