#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
work="$PWD/.prepare/native"
vendor="$PWD/Vendor"
mkdir -p "$work" "$vendor"

fetch() {
    local url=$1 checksum=$2 file=$3
    if ! printf '%s  %s\n' "$checksum" "$file" | shasum -a 256 -c -s 2>/dev/null; then
        curl -fL --retry 3 "$url" -o "$file"
        printf '%s  %s\n' "$checksum" "$file" | shasum -a 256 -c
    fi
}

if [[ ! -f "$vendor/sherpa-onnx.xcframework/Info.plist" ]]; then
    fetch "https://github.com/k2-fsa/sherpa-onnx/releases/download/xcframework/sherpa-onnx-v1.13.4-ios-static.xcframework.zip" \
        b48ec217952a5b82242ce7d8323fcbc8de54ff900a72df1f0b20bfcf7b08881d "$work/sherpa.zip"
    unzip -oq "$work/sherpa.zip" -d "$vendor"
fi
if [[ ! -f "$vendor/onnxruntime.xcframework/Info.plist" ]]; then
    fetch "https://github.com/csukuangfj/onnxruntime-libs/releases/download/v1.27.0/onnxruntime-ios-static-xcframework-1.27.0.zip" \
        2c4b6eda7fcf03ca51814bbc88e3709cc080e623581fce085286182cc30d60c1 "$work/ort.zip"
    unzip -oq "$work/ort.zip" -d "$work"
    mv "$work/onnxruntime-ios-static-xcframework-1.27.0/onnxruntime.xcframework" "$vendor/"
fi
if [[ ! -f "$vendor/SentencePiece.xcframework/Info.plist" ]]; then
    fetch "https://codeload.github.com/google/sentencepiece/tar.gz/refs/tags/v0.2.1" \
        c1a59e9259c9653ad0ade653dadff074cd31f0a6ff2a11316f67bee4189a8f1b "$work/sentencepiece.tar.gz"
    tar xzf "$work/sentencepiece.tar.gz" -C "$work"
    source="$work/sentencepiece-0.2.1"
    # Upstream sets Xcode executable properties via a helper from its optional toolchain.
    sed -i "" 's/^  set_xcode_property.*$/  # Static library build omits executable settings./' "$source/src/CMakeLists.txt"
    mkdir -p "$work/headers"
    cp "$source/src/sentencepiece_processor.h" "$work/headers/"
    for sdk in iphoneos iphonesimulator; do
        cmake -S "$source" -B "$work/$sdk" -DCMAKE_SYSTEM_NAME=iOS \
            -DCMAKE_OSX_SYSROOT="$(xcrun --sdk "$sdk" --show-sdk-path)" \
            -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=17.0 \
            -DCMAKE_BUILD_TYPE=Release -DSPM_ENABLE_SHARED=OFF -DSPM_ENABLE_TCMALLOC=OFF
        cmake --build "$work/$sdk" --target sentencepiece-static --parallel 4
    done
    xcodebuild -create-xcframework \
        -library "$work/iphoneos/src/libsentencepiece.a" -headers "$work/headers" \
        -library "$work/iphonesimulator/src/libsentencepiece.a" -headers "$work/headers" \
        -output "$vendor/SentencePiece.xcframework"
    cp "$source/LICENSE" "$vendor/SentencePiece.LICENSE"
fi
