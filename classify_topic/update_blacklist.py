import os
import re
import csv
import requests
import shutil
import subprocess
import logging
from collections import defaultdict
from tqdm import tqdm  # Progress bar

# --- Configuration ---
DEST_DIR = "./classify_topic/blacklists_curate/"
TEMP_UT1_DIR = "./temp_ut1_download/"
UT1_RSYNC_URL = "rsync://ftp.ut-capitole.fr/blacklist"
CC_SHEET_ID = "12NlJFJJnmzvgXarrF_VqdhVuWtX-k_gtItH80cRkaQ4"
CC_GID = "406780385"
CC_CSV_URL = f"https://docs.google.com/spreadsheets/d/{CC_SHEET_ID}/export?format=csv&gid={CC_GID}"

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blacklist_update.log"),
        logging.StreamHandler()
    ]
)

# --- Pre-compile Regex Patterns for Speed ---
CATEGORY_RULES = [
    ("adult", re.compile(r"sex|porn|adult|xxx|69", re.I)),
    ("malware", re.compile(r"virus|trojan|malware|infect", re.I)),
    ("phishing", re.compile(r"phish|login|verify|secure|account|update", re.I)),
    ("shopping", re.compile(r"shop|store|market|amazon|buy|sale", re.I)),
    ("games", re.compile(r"game|play|steam|xbox|nintendo", re.I)),
    ("publicite", re.compile(r"banner|marketing", re.I)),
    ("press", re.compile(r"publi|news|diari|gazette|libre|journal|magazine|mag|sports|publi|liber|minut|live", re.I)),
    ("radio", re.compile(r"tv|radio|fm", re.I)),
    ("audio-video", re.compile(r"tv", re.I)),
    ("shortener", re.compile(r"short|tinyurl|bitly|goo\.gl", re.I)),
    ("download", re.compile(r"download|file|setup|exe|apk", re.I)),
    ("sports", re.compile(r"sport|football|nba|fifa|tennis", re.I)),
    ("bank", re.compile(r"bank|banco|credit|finance|transfer", re.I)),
    ("blog", re.compile(r"blog|wordpress|medium|notion|diary", re.I)),
    ("filehosting", re.compile(r"upload|filehost|mega|rapidgator|dropbox", re.I)),
]

DEFAULT_CATEGORY = "press"

# Regex to split mashed domains (e.g., .frNext)
SPLIT_PATTERN = re.compile(r'(\.[a-zA-Z]{2,6})(?=[A-Z0-9a-z])')
# Regex to extract valid URLs/Domains
EXTRACT_PATTERN = re.compile(r'(?:https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^,\s"\'<>]*)?')

def get_category(url):
    """Categorizes a URL based on pre-compiled regex."""
    # Note: We don't lower() the URL here to save time, we used re.I (Ignore Case) in compilation
    for category, pattern in CATEGORY_RULES:
        if pattern.search(url):
            return category
    return DEFAULT_CATEGORY

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def read_file_to_set(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return set(line.strip() for line in f if line.strip())

def write_set_to_file(filepath, data_set):
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(data_set)) + '\n')

# --- UT1 Functions (Keep as is, rsync is efficient) ---
def fetch_ut1_rsync():
    logging.info("Starting UT1 Rsync download...")
    ensure_dir(TEMP_UT1_DIR)
    cmd = ["rsync", "-av", "--delete", UT1_RSYNC_URL, TEMP_UT1_DIR]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("UT1 Rsync download successful.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Rsync failed: {e.stderr.decode()}")

def process_ut1_files():
    logging.info("Processing UT1 files...")
    stats = defaultdict(lambda: {"added": 0, "removed": 0})
    for root, dirs, files in os.walk(TEMP_UT1_DIR):
        for file in files:
            if file not in ["domains", "urls"]: continue
            category = os.path.basename(root)
            src_path = os.path.join(root, file)
            dest_path = os.path.join(DEST_DIR, category, file)

            src_set = read_file_to_set(src_path)
            dest_set = read_file_to_set(dest_path)

            if src_set != dest_set:
                write_set_to_file(dest_path, src_set)
                stats[category]["added"] += len(src_set - dest_set)
                stats[category]["removed"] += len(dest_set - src_set)
                logging.info(f"UT1: Updated {category}/{file} - Added: {len(src_set - dest_set)}, Removed: {len(dest_set - src_set)}")
    
    shutil.rmtree(TEMP_UT1_DIR)
    return stats

# --- Optimized CommonCrawl Function ---

def fetch_and_process_cc():
    """
    Downloads CC CSV, batches updates in memory, and writes to disk once per file.
    """
    logging.info("Downloading CommonCrawl CSV...")
    try:
        response = requests.get(CC_CSV_URL)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download CC CSV: {e}")
        return

    # Read CSV content
    decoded_content = response.content.decode('utf-8')
    rows = list(csv.reader(decoded_content.splitlines(), delimiter=','))
    
    # In-Memory Batch Storage
    # Structure: batch_data[category][file_type] = set([url1, url2])
    batch_data = defaultdict(lambda: defaultdict(set))
    
    logging.info("Parsing CommonCrawl Data...")
    
    # TQDM Progress Bar over rows
    for row in tqdm(rows, desc="Processing Rows", unit="row"):
        if not row or len(row) < 3: continue
        if "List of domains" in row[2]: continue

        raw_text = row[2]
        
        # 1. Clean "mashed" domains
        cleaned_text = SPLIT_PATTERN.sub(r'\1 ', raw_text)
        
        # 2. Extract URLs
        extracted_items = EXTRACT_PATTERN.findall(cleaned_text)

        for item in extracted_items:
            clean_item = item.rstrip('.,/)')
            if len(clean_item) < 4 or "." not in clean_item: continue

            category = get_category(clean_item)
            
            # Determine type
            item_no_proto = re.sub(r'^https?://', '', clean_item)
            file_type = "urls" if "/" in item_no_proto else "domains"
            
            # Add to memory batch (Instant)
            batch_data[category][file_type].add(clean_item)

    # --- Write Phase ---
    logging.info("Writing updates to disk...")
    total_added = 0
    
    # Iterate only through categories that have updates
    for category, file_types in tqdm(batch_data.items(), desc="Updating Files", unit="cat"):
        for file_type, new_urls_set in file_types.items():
            
            dest_path = os.path.join(DEST_DIR, category, file_type)
            
            # Read existing disk data ONCE
            current_disk_set = read_file_to_set(dest_path)
            
            # Find truly new items
            truly_new = new_urls_set - current_disk_set
            
            if truly_new:
                # Union and Write back ONCE
                combined_set = current_disk_set.union(truly_new)
                write_set_to_file(dest_path, combined_set)
                
                count = len(truly_new)
                total_added += count
                logging.info(f"CC: Added {count} entries to {category}/{file_type}")

    logging.info(f"CommonCrawl Update Complete. Total new entries: {total_added}")

def main():
    logging.info("--- Starting Blacklist Update ---")
    
    # 1. UT1
    try:
        fetch_ut1_rsync()
        process_ut1_files()
    except Exception as e:
        logging.error(f"UT1 Error: {e}")

    # 2. CommonCrawl
    try:
        fetch_and_process_cc()
    except Exception as e:
        logging.error(f"CC Error: {e}")

    logging.info("--- All Updates Completed ---")

if __name__ == "__main__":
    main()