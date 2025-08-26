from utils_extract_features import *
import logging
import time
import sys

logging.basicConfig(level=logging.INFO)

BURST_WINDOW_SECONDS = 1.0

def extract_features(pcap_file: str, labels: Dict[str, int], dflag: bool = False) -> Dict[str, any]:
    """
    Extract network traffic features from a PCAP file.
    
    Args:
        pcap_file: Path to the PCAP file
        labels: Dictionary mapping device labels to numerical classes
        dflag: Debug flag to print processing time
        
    Returns:
        Dictionary of extracted features
    """
    if dflag:
        start_time = time.time()
    
    packet_stats = {
        'packet_times': [],
        'packet_sizes': [],
        'protocols': set(),
        'http_methods': set(),
        'tls_versions': set(),
        'tls_ciphers': set(),
        
        'total_outgoing': 0,
        'total_inbound': 0,
        'outgoing_lengths': [],
        'inbound_lengths': [],
        'inbound_timestamps': [],
        
        'out_flags': {'syn': 0, 'ack': 0, 'psh': 0, 'fin': 0, 'urg': 0, 'rst': 0},
        'in_flags': {'syn': 0, 'ack': 0, 'psh': 0, 'fin': 0, 'urg': 0, 'rst': 0},
        
        'protocol_counts': {'in': defaultdict(int), 'out': defaultdict(int)},
        
        'tcp_connections': defaultdict(int),
    }
    
    try:     
        packets = rdpcap(pcap_file)  
    
    except Exception as e:
        logging.error(f"Error loading PCAP: {e}, file: {pcap_file}")
        return {}
        
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
            
        ts = float(pkt.time)
        size = len(pkt)
        proto = pkt.lastlayer().name
        
        packet_stats['packet_times'].append(ts)
        packet_stats['packet_sizes'].append(size)
        packet_stats['protocols'].add(proto)
        
        is_outgoing = not pkt[IP].src.endswith(SERVER_IP_SUFFIX)
        
        if is_outgoing:
            process_outgoing_packet(pkt, packet_stats, size, proto, ts)
        else:
            process_inbound_packet(pkt, packet_stats, size, proto, ts)

    
    if not packet_stats['packet_times']:
        logging.warning("No valid packets found in the capture")
        return {}
    
    packet_times_np = np.array(packet_stats['packet_times'])
    packet_sizes_np = np.array(packet_stats['packet_sizes'])
    inbound_timestamps_np = np.array(packet_stats['inbound_timestamps']) if packet_stats['inbound_timestamps'] else np.array([])
    
    total_packet_count = len(packet_sizes_np)
    protocols_list = list(packet_stats['protocols'])
    label, update_name = get_label_from_filename(pcap_file, labels)
    
    duration = packet_times_np.max() - packet_times_np.min() if len(packet_times_np) > 1 else 0
    
    features = {}
    
    features.update(compute_basic_stats(packet_stats, label, total_packet_count, 
                                        protocols_list, duration))
    
    features.update(calculate_interarrival_metrics(packet_times_np))
    features.update(calculate_interarrival_metrics(inbound_timestamps_np, prefix='in_'))
    
    features.update(calculate_length_statistics(packet_stats['inbound_lengths'], 'in'))
    features.update(calculate_length_statistics(packet_stats['outgoing_lengths'], 'out'))
    
    features.update(calculate_flag_percentages(
        packet_stats['out_flags'], 
        packet_stats['total_outgoing'], 
        'out'
    ))
    features.update(calculate_flag_percentages(
        packet_stats['in_flags'], 
        packet_stats['total_inbound'], 
        'in'
    ))
    
    features['in_max_burstnumpkts'] = calculate_max_burst(
        inbound_timestamps_np, 
        BURST_WINDOW_SECONDS
    )

    features['out_max_burstnumpkts'] = calculate_max_burst(
        packet_times_np, 
        BURST_WINDOW_SECONDS
    )

    features['update_name'] = update_name
    
    if dflag:
        processing_time = time.time() - start_time
        logging.info(f"Feature extraction completed in {processing_time:.2f} seconds")
    
    return features


############################### This is just a test function to run the script directly ###############################
if __name__ == "__main__":
 
    if len(sys.argv) < 2:
        print("Usage: python clean_extract_features.py [pcap_file] [label_map_file]")
        sys.exit(1)
    
    pcap_file = sys.argv[1]
    label_map_file = sys.argv[2] if len(sys.argv) > 2 else "label_map.txt"
    
    labels = load_label_map(label_map_file)
    
    print(f"Extracting features from {pcap_file}")
    start = time.time()
    features = extract_features(pcap_file, labels)
    end = time.time()
    
    print("=" * 50)
    print(f"Extracted {len(features)} features in {end-start:.2f} seconds")
    print(features)
    print("=" * 50)
    
    # Print feature keys in a more readable format
    print("Feature categories:")
    categories = {
        "Basic": ["label", "total_packet_count", "duration"],
        "Protocol": ["tcp_count", "tls_count", "tcp_connections"],
        "Size": ["total_bytes_sent", "total_bytes_received", "avg_packet_size"],
        "Timing": [k for k in features.keys() if "inter_arrival" in k],
        "Direction": ["inbound_outbound_ratio", "out_percentage", "in_percentage"],
        "TCP Flags": [k for k in features.keys() if "percentage" in k and k.split("_")[1] in 
                     ["rst", "psh", "fin", "urg", "syn", "ack"]],
        "Length Stats": [k for k in features.keys() if k.endswith("_len")]
    }
    
    for category, keys in categories.items():
        present_keys = [k for k in keys if k in features]
        if present_keys:
            print(f"  {category}: {len(present_keys)} features")
    
    print("=" * 50)