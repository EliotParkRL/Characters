import csv

# Input and output file paths
input_csv_path = "/Users/eliotpark/Desktop/GitHub/Characters/labels.csv"
output_csv_path = "/Users/eliotpark/Desktop/GitHub/Characters/processed_labels.csv"

# Process the CSV file
with open(input_csv_path, mode="r") as infile, open(output_csv_path, mode="w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Process each row in the input CSV
    for row in reader:
        # Modify the image path format
        image_id = row[0].replace("Img/img", "").replace(".png", "")
        label = row[1]
        
        # Write the modified row to the output CSV
        writer.writerow([image_id, label])

print(f"Formatted CSV has been saved to {output_csv_path}")
