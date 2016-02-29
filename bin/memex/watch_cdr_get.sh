#!/usr/bin/env bash

output_dir="test_output"
output_csv="test_output.csv"
log_file="test_output.log"
interval=1  # seconds

watch -n ${interval} "
    echo \"CDR records scanned             : \$(grep \"report_progress\" \"${log_file}\" | tail -1 | cut -d' ' -f20)\";
    echo \"Files Downloaded                : \$(find \"${output_dir}/\" -type f | wc -l)\";
    echo \"Files Recorded                  : \$(cat \"${output_csv}\" | wc -l)\";
    echo \"Download stored-data failures   : \$(grep \"stored-data URL\" \"${log_file}\" | wc -l)\";
    echo \"Download fallback failures      : \$(grep \"original URL\" \"${log_file}\" | wc -l)\";
    echo \"DL Failures caused by exceptions: \$(grep \"\(Exception\|Error\|FAILED\)\" \"${log_file}\" | wc -l)\"
    echo \"Scan Restarts                   : \$(grep \"Restarting query\" \"${log_file}\" | wc -l)\"
"
