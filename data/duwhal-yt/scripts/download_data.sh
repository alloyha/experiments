#!/usr/bin/env bash

set -euo pipefail


PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"

ARCHIVE="${DATA_DIR}/KuaiRec.zip"
TARGET_DIR="${DATA_DIR}/KuaiRec"
EXTRACTED_DIR="${DATA_DIR}/KuaiRec 2.0"

URL="https://zenodo.org/records/18164998/files/KuaiRec.zip"


mkdir -p "${DATA_DIR}"

cd "${DATA_DIR}"


if [[ -d "${TARGET_DIR}" ]]; then
    echo "KuaiRec already exists at:"
    echo "${TARGET_DIR}"
    exit 0
fi


if [[ ! -f "${ARCHIVE}" ]]; then
    echo "Downloading KuaiRec..."

    wget \
        "${URL}" \
        -O "${ARCHIVE}"
else
    echo "Using existing archive:"
    echo "${ARCHIVE}"
fi


echo "Extracting..."

rm -rf "${EXTRACTED_DIR}"

unzip -o "${ARCHIVE}"


if [[ ! -d "${EXTRACTED_DIR}" ]]; then
    echo "Expected extracted directory not found:"
    echo "${EXTRACTED_DIR}"
    exit 1
fi


mv "${EXTRACTED_DIR}" "${TARGET_DIR}"

rm -f "${ARCHIVE}"


echo
echo "KuaiRec installed at:"
echo "${TARGET_DIR}"

echo
echo "Dataset files:"
ls -lh "${TARGET_DIR}/data"

