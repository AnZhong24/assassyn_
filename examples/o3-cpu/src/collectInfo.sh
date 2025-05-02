log_directory="."
output_csv="output.csv"  # Define output CSV file path
echo "base_name,total_counts_str,total_wrong_counts_str,predict_wrong_count,not_issued_count,Oldest_count,issue_direct_count,newinst_count,last_cycle,total_3_count,total_4_count,total_5_count,total_6_count,total_wrong_1_count,total_wrong_2_count" > "$output_csv"  # Write CSV header

# Iterate over all .log files
for log_file in "$log_directory"/*.log
do
    if [ -f "$log_file" ]; then
        # Get the base name (without path)
        base_name=$(basename "$log_file")
        
        # --- Calculate predict_wrong count ---
        predict_wrong_count=$(grep -Fo "wrong" "$log_file" | wc -l)
        issue_direct_count=$(grep -Fo "Dispatch" "$log_file" | wc -l)
        not_issued_count=$(grep -Fo "DIRECT" "$log_file" | wc -l)
        newinst_count=$(grep -Fo "INDIRE" "$log_file" | wc -l)
        Oldest_count=$(grep -Fo "NOISSUE" "$log_file" | wc -l)

        # --- Count occurrences of Total=3, Total=4, Total=5, Total_Wrong=1, Total_Wrong=2 ---
        total_3_count=$(grep -Fo "Total=3" "$log_file" | wc -l)
        total_4_count=$(grep -Fo "Total=4" "$log_file" | wc -l)
        total_5_count=$(grep -Fo "Total=5" "$log_file" | wc -l)
	total_6_count=$(grep -Fo "Total=6" "$log_file" | wc -l)
	total_7_count=$(grep -Fo "Total=7" "$log_file" | wc -l)

	total_wrong_1_count=$(grep -Fo "Total_Wrong=1" "$log_file" | wc -l)
        total_wrong_2_count=$(grep -Fo "Total_Wrong=2" "$log_file" | wc -l)
	total_wrong_4_count=$(grep -Fo "Total_Wrong=4" "$log_file" | wc -l)

	total_wrong_5_count=$(grep -Fo "Total_Wrong=5" "$log_file" | wc -l)
	total_wrong_3_count=$(grep -Fo "Total_Wrong=3" "$log_file" | wc -l)
        # --- Find last cycle line and extract cycle number ---
        last_cycle="N/A"  # Default value
        extracted_cycle=""  # Clear previous value

        # Find the last line containing "Cycle @"
        last_cycle_line=$(grep "Cycle @" "$log_file" | tail -n 1)

        # Check if a line was actually found
        if [[ -n "$last_cycle_line" ]]; then
            # Extract the cycle number using grep with Perl-compatible regex (-P)
            extracted_cycle=$(echo "$last_cycle_line" | grep -oP 'Cycle @\K[0-9]+')

            # Basic validation: check if extracted value consists ONLY of digits
            if [[ "$extracted_cycle" =~ ^[0-9]+$ ]]; then
                last_cycle="$extracted_cycle"
            else
                echo "    Warning: Could not extract valid cycle number from last 'Cycle @' line in $log_file. Line: '$last_cycle_line'. Extracted: '$extracted_cycle'"
                # Keep last_cycle as "N/A"
            fi
        else
            echo "    Info: No 'Cycle @' line found in $log_file."
            # Keep last_cycle as "N/A"
        fi

        # --- Format and write the CSV row ---
        # Append the base filename and all calculated metrics to the CSV file
        #echo "$base_name,$total_counts_str,$total_wrong_counts_str,$predict_wrong_count,$not_issued_count,$Oldest_count,$issue_direct_count,$newinst_count,$last_cycle,$total_3_count,$total_4_count,$total_5_count,$total_6_count,$total_7_count,$total_wrong_1_count,$total_wrong_2_count,$total_wrong_3_count,$total_wrong_4_count,$total_wrong_5_count" >> "$output_csv"
	
	 echo "$base_name,$not_issued_count,$newinst_count,$last_cycle" >> "$output_csv"
    fi
done

shopt -u nullglob  # Turn off nullglob if needed elsewhere

echo "Processing complete. Results saved to $output_csv"

