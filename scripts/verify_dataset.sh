#!/bin/bash

# Verify GCS dataset parquet files
set -e

# Export variables from .env file
set -o allexport
source ../.neobert.env
set +o allexport

echo "=== NeoBERT Dataset Verification ==="
echo "Training dataset path: ${DATASET_PATH}"
echo "Validation dataset path: ${VALIDATION_DATASET_PATH}"
echo "Text column: ${TEXT_COLUMN}"
echo ""

# Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil is not installed or not in PATH"
    exit 1
fi

# Check training dataset
echo "=== Training Dataset Check ==="
echo "Listing parquet files in ${DATASET_PATH}..."
PARQUET_FILES=$(gsutil ls "${DATASET_PATH}/*.parquet" 2>/dev/null || echo "")

if [ -z "$PARQUET_FILES" ]; then
    echo "ERROR: No parquet files found in ${DATASET_PATH}"
    echo "Please check that:"
    echo "1. The DATASET_PATH is correct in .neobert.env"
    echo "2. The GCS bucket is accessible"
    echo "3. Parquet files exist in the specified path"
    exit 1
fi

echo "Found training parquet files:"
echo "$PARQUET_FILES" | head -10
TOTAL_FILES=$(echo "$PARQUET_FILES" | wc -l)
echo "Total training files: $TOTAL_FILES"

if [ $TOTAL_FILES -gt 10 ]; then
    echo "(showing first 10 files only)"
fi

echo ""

# Check validation dataset
echo "=== Validation Dataset Check ==="
if [ -n "$VALIDATION_DATASET_PATH" ]; then
    echo "Listing validation parquet files in ${VALIDATION_DATASET_PATH}..."
    VAL_PARQUET_FILES=$(gsutil ls "${VALIDATION_DATASET_PATH}/*.parquet" 2>/dev/null || echo "")

    if [ -z "$VAL_PARQUET_FILES" ]; then
        echo "WARNING: No validation parquet files found in ${VALIDATION_DATASET_PATH}"
        echo "Training will continue without validation"
    else
        echo "Found validation parquet files:"
        echo "$VAL_PARQUET_FILES"
        TOTAL_VAL_FILES=$(echo "$VAL_PARQUET_FILES" | wc -l)
        echo "Total validation files: $TOTAL_VAL_FILES"
    fi
else
    echo "No validation dataset path specified - training will run without validation"
    VAL_PARQUET_FILES=""
fi

echo ""

# Download and inspect the first training parquet file
FIRST_FILE=$(echo "$PARQUET_FILES" | head -1)
echo "Downloading first training file for inspection: $(basename $FIRST_FILE)"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
LOCAL_FILE="$TEMP_DIR/sample.parquet"

gsutil cp "$FIRST_FILE" "$LOCAL_FILE"

echo ""
echo "=== Training File Structure Analysis ==="

# Use Python to analyze the parquet file
python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_parquet('$LOCAL_FILE')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("")

    # Check if text column exists
    if '$TEXT_COLUMN' in df.columns:
        print(f"✓ Text column '{TEXT_COLUMN}' found")

        # Sample some text entries
        text_series = df['$TEXT_COLUMN'].dropna()
        if len(text_series) > 0:
            print(f"Non-null text entries: {len(text_series)}")
            print(f"Sample text lengths: {text_series.str.len().describe().to_dict()}")
            print("")
            print("Sample texts:")
            for i, text in enumerate(text_series.head(3)):
                print(f"  {i+1}. {str(text)[:100]}{'...' if len(str(text)) > 100 else ''}")
        else:
            print(f"⚠️  WARNING: No non-null entries in '{TEXT_COLUMN}' column")
    else:
        print(f"✗ ERROR: Text column '{TEXT_COLUMN}' not found")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

except Exception as e:
    print(f"ERROR analyzing parquet file: {e}")
    sys.exit(1)
EOF

# Check validation file if it exists
if [ -n "$VAL_PARQUET_FILES" ]; then
    FIRST_VAL_FILE=$(echo "$VAL_PARQUET_FILES" | head -1)
    echo ""
    echo "=== Validation File Structure Analysis ==="
    echo "Downloading first validation file for inspection: $(basename $FIRST_VAL_FILE)"

    LOCAL_VAL_FILE="$TEMP_DIR/sample_val.parquet"
    gsutil cp "$FIRST_VAL_FILE" "$LOCAL_VAL_FILE"

    python3 << EOF
import pandas as pd
import sys

try:
    df = pd.read_parquet('$LOCAL_VAL_FILE')
    print(f"Validation shape: {df.shape}")
    print(f"Validation columns: {list(df.columns)}")
    print("")

    # Check if text column exists
    if '$TEXT_COLUMN' in df.columns:
        print(f"✓ Validation text column '{TEXT_COLUMN}' found")

        # Sample some text entries
        text_series = df['$TEXT_COLUMN'].dropna()
        if len(text_series) > 0:
            print(f"Validation non-null text entries: {len(text_series)}")
            print(f"Validation sample text lengths: {text_series.str.len().describe().to_dict()}")
            print("")
            print("Validation sample texts:")
            for i, text in enumerate(text_series.head(2)):
                print(f"  {i+1}. {str(text)[:100]}{'...' if len(str(text)) > 100 else ''}")
        else:
            print(f"⚠️  WARNING: No non-null entries in validation '{TEXT_COLUMN}' column")
    else:
        print(f"✗ ERROR: Validation text column '{TEXT_COLUMN}' not found")
        print(f"Available validation columns: {list(df.columns)}")
        sys.exit(1)

except Exception as e:
    print(f"ERROR analyzing validation parquet file: {e}")
    sys.exit(1)
EOF

    # Check Python exit code for validation
    if [ $? -ne 0 ]; then
        echo ""
        echo "Validation dataset verification FAILED"
        rm -rf "$TEMP_DIR"
        exit 1
    fi
fi

# Check Python exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "Dataset verification FAILED"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Cleanup
rm -rf "$TEMP_DIR"

echo ""
echo "=== Configuration Verification ==="
echo "✓ Training GCS path accessible"
echo "✓ Training parquet files found ($TOTAL_FILES files)"
echo "✓ Training text column '${TEXT_COLUMN}' exists"
echo "✓ Training sample data looks valid"

if [ -n "$VAL_PARQUET_FILES" ]; then
    echo "✓ Validation GCS path accessible"
    echo "✓ Validation parquet files found ($TOTAL_VAL_FILES files)"
    echo "✓ Validation text column '${TEXT_COLUMN}' exists"
    echo "✓ Validation sample data looks valid"
fi

echo ""
echo "=== Recommendations ==="
if [ $TOTAL_FILES -gt 100 ]; then
    echo "• Large dataset detected ($TOTAL_FILES files) - consider using streaming mode"
    echo "• Set STREAMING=true in .neobert.env for memory efficiency"
fi

if [ $TOTAL_FILES -lt 10 ]; then
    echo "• Small dataset detected ($TOTAL_FILES files) - consider using in-memory loading"
    echo "• Set STREAMING=false in .neobert.env for better performance"
fi

echo "• Recommended STEPS_PER_EPOCH: $(($TOTAL_FILES * 1000 / 8))"  # Rough estimate
if [ -n "$VAL_PARQUET_FILES" ]; then
    echo "• Validation will run every ${EVAL_EVERY_N_EPOCHS:-1} epoch(s)"
    echo "• Validation steps per evaluation: ${VALIDATION_STEPS:-100}"
else
    echo "• No validation will be performed (consider adding VALIDATION_DATASET_PATH)"
fi
echo "• Verify tokenizer path: ${NEOBERT_TOKENIZER_PATH}"

echo ""
echo "Dataset verification PASSED ✓"
if [ -n "$VAL_PARQUET_FILES" ]; then
    echo "Ready for NeoBERT training with validation!"
else
    echo "Ready for NeoBERT training (no validation)!"
fi
