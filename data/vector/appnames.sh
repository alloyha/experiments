#!/bin/bash

logfile="$1"

# Clear all files with "*_timestamps.txt" extension
find . -name "*_timestamps.txt" -type f -delete

if [ -f "invalid_lines.txt" ]; then
    rm 'invalid_lines.txt'
fi

# Define the function to process a single line
process_line() {
    line="$1"
    appname=$(echo "$line" | awk -F': ' '{print $NF}' | jq -r '.appname' 2>/dev/null)
    timestamp=$(echo "$line" | awk -F': ' '{print $NF}' | jq -r '.timestamp' 2>/dev/null)

    if [ -z "$appname" ] || [ -z "$timestamp" ]; then
        echo "$line" >> invalid_lines.txt  # Append the invalid line to a file
        return 1  # Skip invalid lines
    fi
    
    filename="${appname}_timestamps.txt"
    echo "$timestamp" >> "$filename"
}

export -f process_line  # Export the function to make it available to parallel

# Use GNU Parallel to process lines in parallel and display ETA
cat "$logfile" | parallel --eta process_line {}

# Calculate the expected time to finish all tasks
total_lines=$(wc -l < "$logfile")

# Display summary
echo "Total lines: $total_lines"

echo "Processed files:"
while IFS= read -r file; do
    line_count=$(wc -l < "$file")
    echo "$file - $line_count lines"
done < <(find . -name "*_timestamps.txt")

invalid_lines=0
if [ -f "invalid_lines.txt" ]; then
    invalid_lines=$(wc -l < invalid_lines.txt)
fi
echo "Invalid lines: $invalid_lines lines"



