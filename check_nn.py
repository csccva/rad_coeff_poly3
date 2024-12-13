import re

# Open the file and read its contents
with open('nn.txt', 'r') as file:
    text = file.read()

# Define regular expressions for matching the relevant lines
buffer_region_pattern = r"Buffer_Region_site\s+(\d+)\s+nn\s+is\s+(\d+)"
site_thread_pattern = r"Site\s+(\d+)\s+thread\s+\d+\s+nn\s+(\d+)\s+broadcasted\s+nn\s+(\d+)"

# Initialize dictionaries to store values for comparison
buffer_values = {}
site_values = {}

# Find all matches for Buffer_Region_site lines
buffer_matches = re.findall(buffer_region_pattern, text)
for s, n in buffer_matches:
    buffer_values[int(s)] = int(n)

# Find all matches for Site thread nn broadcasted nn lines
site_matches = re.findall(site_thread_pattern, text)
for s, nn1, nn2 in site_matches:
    site_values[int(s)] = (int(nn1), int(nn2))

# Initialize a counter for non-matching lines
non_matching_count = 0

# Compare the values for the same value of S
for s in buffer_values:
    if s in site_values:
        buffer_n = buffer_values[s]
        site_nn1, site_nn2 = site_values[s]
        print(f"Site {s}:")
        print(f"  Buffer Region nn: {buffer_n}, Site nn1: {site_nn1}, Site nn2: {site_nn2}")
        print(f"  Difference in nn: {abs(buffer_n - site_nn1)}, {abs(buffer_n - site_nn2)}\n")
        
        # Check for mismatches
        if buffer_n != site_nn1 or buffer_n != site_nn2:
            non_matching_count += 1
    else:
        # If the site value is missing, count it as a mismatch
        non_matching_count += 1

# Print the number of non-matching lines
print(f"Number of non-matching lines: {non_matching_count}")
