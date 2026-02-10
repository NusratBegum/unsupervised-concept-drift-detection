"""Script to add proper headers to the USP DS Repository CSV files."""
import os

BASE_PATH = "datasets/files"

# Define the header configurations for each dataset type
DATASETS = {
    # INSECTS datasets: 33 attributes (Att1-Att33) + class
    "INSECTS-abrupt_balanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-abrupt_imbalanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-gradual_balanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-gradual_imbalanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-incremental_balanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-incremental_imbalanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-incremental-abrupt_balanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-incremental-abrupt_imbalanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-incremental-reoccurring_balanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    "INSECTS-incremental-reoccurring_imbalanced_norm.csv": {"attrs": [f"Att{i}" for i in range(1, 34)], "target": "class"},
    
    # NOAA Weather: 8 attributes + class
    "NOAA.csv": {"attrs": [f"attribute{i}" for i in range(1, 9)], "target": "class"},
    
    # Outdoor Objects: 21 attributes + class
    "outdoor.csv": {"attrs": [f"att{i}" for i in range(1, 22)], "target": "class"},
    
    # Luxembourg: 31 attributes + class
    "luxembourg.csv": {"attrs": [f"att{i}" for i in range(1, 32)], "target": "class"},
    
    # Powersupply: 2 attributes + class
    "powersupply.csv": {"attrs": ["attribute0", "attribute1"], "target": "class"},
    
    # Ozone: 72 attributes + class
    "ozone.csv": {"attrs": [f"att{i}" for i in range(1, 73)], "target": "class"},
    
    # Rialto: 27 attributes + class
    "rialto.csv": {"attrs": [f"att{i}" for i in range(1, 28)], "target": "class"},
    
    # Poker Hand: 10 attributes (s1,c1,s2,c2,...,s5,c5) + class
    "poker-lsn.csv": {"attrs": ["s1", "c1", "s2", "c2", "s3", "c3", "s4", "c4", "s5", "c5"], "target": "class"},
    
    # Sensor Stream: many attributes
    "sensorstream.csv": {"attrs": ["rcdminutes"] + [f"att{i}" for i in range(1, 55)], "target": "class"},
    
    # Electricity: 8 attributes + class (uses .arff format in original, but we have CSV)
    "elec.csv": {"attrs": ["date", "day", "period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"], "target": "class"},
    
    # Forest Covertype: 54 attributes + class
    "covtype.csv": {"attrs": [f"att{i}" for i in range(1, 55)], "target": "class"},
}


def add_header_to_csv(filepath, header):
    """Add a header row to a CSV file."""
    # Read existing content
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check if file already has this header
    first_line = content.split('\n')[0]
    if first_line == header:
        print(f"  {os.path.basename(filepath)}: Already has header, skipping")
        return
    
    # Write header + content
    with open(filepath, 'w') as f:
        f.write(header + '\n' + content)
    
    print(f"  {os.path.basename(filepath)}: Added header")


def main():
    print("Adding headers to USP DS Repository CSV files...\n")
    
    for filename, config in DATASETS.items():
        filepath = os.path.join(BASE_PATH, filename)
        
        if not os.path.exists(filepath):
            print(f"  {filename}: File not found, skipping")
            continue
        
        # Build header string
        header = ",".join(config["attrs"] + [config["target"]])
        add_header_to_csv(filepath, header)
    
    print("\nDone! You can now run the tests.")


if __name__ == "__main__":
    main()
