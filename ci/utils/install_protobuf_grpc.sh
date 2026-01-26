#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Install Protobuf and gRPC C++ development libraries
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "rocky" ]]; then
        echo "Detected Rocky Linux. Installing Protobuf + gRPC via dnf..."
        # Enable PowerTools (Rocky 8) or CRB (Rocky 9) repository for protobuf-devel
        if [[ "${VERSION_ID%%.*}" == "8" ]]; then
            dnf config-manager --set-enabled powertools || dnf config-manager --set-enabled PowerTools || true
        elif [[ "${VERSION_ID%%.*}" == "9" ]]; then
            dnf config-manager --set-enabled crb || true
        fi
        # Protobuf (headers + protoc)
        dnf install -y protobuf-devel protobuf-compiler

        # gRPC C++ (headers/libs + grpc_cpp_plugin for codegen)
        # Package names can vary by repo; try the common ones first.
        dnf install -y grpc-devel grpc-plugins || dnf install -y grpc-devel || true
    elif [[ "$ID" == "ubuntu" ]]; then
        echo "Detected Ubuntu. Installing Protobuf + gRPC via apt..."
        apt-get update
        # Protobuf (headers + protoc)
        apt-get install -y libprotobuf-dev protobuf-compiler

        # gRPC C++ (headers/libs + grpc_cpp_plugin for codegen)
        apt-get install -y libgrpc++-dev protobuf-compiler-grpc
    else
        echo "Unknown OS: $ID. Please install Protobuf + gRPC development libraries manually."
        exit 1
    fi
else
    echo "/etc/os-release not found. Cannot determine OS. Please install Protobuf + gRPC development libraries manually."
    exit 1
fi

