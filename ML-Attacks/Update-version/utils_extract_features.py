import numpy as np
import os
import logging
from typing import Dict, List, Optional, Tuple
from scapy.all import TCP, IP, rdpcap
from collections import defaultdict

logging.basicConfig(level=logging.INFO)

LABEL_MAP_FILE = "label_map.txt"
DEFAULT_LABEL = -1
SERVER_IP_SUFFIX = '.1'

def load_label_map(label_map_file: str = LABEL_MAP_FILE) -> Dict[str, int]:
    """Load mapping from device labels to numerical classes."""
    label_map = {}
    
    try:
        with open(label_map_file, "r") as f:
            for line in f:
                if ':' in line:
                    label, idx = line.strip().split(':')
                    label_map[label.strip()] = int(idx.strip())
    
    except FileNotFoundError:
        logging.warning(f"Label map file not found: {label_map_file}")
    
    return label_map


def get_label_from_filename(pcap_filename: str, label_map: Dict[str, int]) -> int:
    """Extract label from PCAP filename using the mapping."""
    basename = os.path.basename(pcap_filename)
    label_candidate = basename.split("_")[0]
   
    return label_map.get(label_candidate, DEFAULT_LABEL), label_candidate


def calculate_percentage(count: int, total: int) -> float:
    """Calculate percentage safely."""
    return (count / total * 100) if total > 0 else 0.0


def calculate_max_burst(timestamps: np.ndarray, window_size: float) -> int:
    """Calculate maximum burst of packets within a time window."""
    if len(timestamps) == 0:
        return 0
        
    sorted_times = np.sort(timestamps)
    max_burst_count = 0
    start_idx = 0
    
    for i in range(len(sorted_times)):
        while sorted_times[i] - sorted_times[start_idx] >= window_size:
            start_idx += 1
        
        max_burst_count = max(max_burst_count, i - start_idx + 1)
    
    return max_burst_count


def calculate_interarrival_metrics(timestamps: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Calculate interarrival time statistics."""
    metrics = {}
    
    if len(timestamps) > 1:
        sorted_times = np.sort(timestamps)
        inter_arrivals = np.diff(sorted_times)
        
        metrics = {
            f'{prefix}inter_arrival_mean': np.mean(inter_arrivals),
            f'{prefix}inter_arrival_std': np.std(inter_arrivals),
            f'{prefix}inter_arrival_min': np.min(inter_arrivals),
            f'{prefix}inter_arrival_max': np.max(inter_arrivals)
        }
    else:
        metrics = {
            f'{prefix}inter_arrival_mean': 0.0,
            f'{prefix}inter_arrival_std': 0.0,
            f'{prefix}inter_arrival_min': 0.0,
            f'{prefix}inter_arrival_max': 0.0
        }
    
    return metrics


def calculate_length_statistics(lengths: List[int], prefix: str) -> Dict[str, float]:
    """Calculate statistics for packet lengths."""
    if not lengths:
        return {
            f'{prefix}_{stat}_len': None 
            for stat in ['mean', 'min', 'max', 'median', 'std', 'len', '25per', '75per', '10per', '90per']
        }
    
    lengths_np = np.array(lengths)
    
    return {
        f'{prefix}_mean_len': np.mean(lengths_np),
        f'{prefix}_min_len': np.min(lengths_np),
        f'{prefix}_max_len': np.max(lengths_np),
        f'{prefix}_median_len': np.median(lengths_np),
        f'{prefix}_std_len': np.std(lengths_np),
        f'{prefix}_len_len': len(lengths_np),
        f'{prefix}_25per_len': np.percentile(lengths_np, 25),
        f'{prefix}_75per_len': np.percentile(lengths_np, 75),
        f'{prefix}_10per_len': np.percentile(lengths_np, 10),
        f'{prefix}_90per_len': np.percentile(lengths_np, 90),
    }


def calculate_flag_percentages(flag_counts: Dict[str, int], total: int, prefix: str) -> Dict[str, float]:
    """Calculate percentages for TCP flags."""
    return {
        f'{prefix}_{flag}_percentage': calculate_percentage(count, total)
        for flag, count in flag_counts.items()
    }

def process_outgoing_packet(pkt, stats, size, proto, ts):
    """Process an outgoing packet and update statistics."""
    stats['total_outgoing'] += 1
    stats['outgoing_lengths'].append(size)
    stats['protocol_counts']['out'][proto] += 1
    
    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        ip_layer = pkt[IP]
        
        src_ip, dst_ip = ip_layer.src, ip_layer.dst
        sport, dport = tcp_layer.sport, tcp_layer.dport
        conn_key = tuple(sorted((src_ip, dst_ip)) + sorted((sport, dport)))
        stats['tcp_connections'][conn_key] += 1
        
        process_flags(tcp_layer, stats['out_flags'])


def process_inbound_packet(pkt, stats, size, proto, ts):
    """Process an inbound packet and update statistics."""
    stats['total_inbound'] += 1
    stats['inbound_lengths'].append(size)
    stats['inbound_timestamps'].append(ts)
    stats['protocol_counts']['in'][proto] += 1
    
    if pkt.haslayer(TCP):
        tcp_layer = pkt[TCP]
        process_flags(tcp_layer, stats['in_flags'])


def process_flags(tcp_layer, flag_counter):
    """Extract and count TCP flags."""
    if tcp_layer.flags:
        flags = tcp_layer.flags

        flag_mapping = [
            (0x04, 'rst'),  # Reset
            (0x08, 'psh'),  # Push
            (0x01, 'fin'),  # Finish
            (0x20, 'urg'),  # Urgent
            (0x02, 'syn'),  # Synchronize
            (0x10, 'ack')   # Acknowledge
        ]
        
        for bit, name in flag_mapping:
            if flags & bit:
                flag_counter[name] += 1


def compute_basic_stats(stats, label, total_packet_count, protocols_list, duration):
    """Compute basic statistical features."""
    
    outgoing_lengths_np = np.array(stats['outgoing_lengths']) if stats['outgoing_lengths'] else np.array([])
    inbound_lengths_np = np.array(stats['inbound_lengths']) if stats['inbound_lengths'] else np.array([])
    packet_sizes_np = np.array(stats['packet_sizes']) if stats['packet_sizes'] else np.array([])
    
    total_outgoing = stats['total_outgoing']
    total_inbound = stats['total_inbound']
    
    return {
        'label': label,
        'total_packet_count': total_packet_count,
        'total_bytes_sent': np.sum(outgoing_lengths_np) if len(outgoing_lengths_np) > 0 else 0,
        'total_bytes_received': np.sum(inbound_lengths_np) if len(inbound_lengths_np) > 0 else 0,
        'avg_packet_size': np.mean(packet_sizes_np) if len(packet_sizes_np) > 0 else 0,
        'min_packet_size': np.min(packet_sizes_np) if len(packet_sizes_np) > 0 else 0,
        'max_packet_size': np.max(packet_sizes_np) if len(packet_sizes_np) > 0 else 0,
        'packet_size_variance': np.var(packet_sizes_np) if len(packet_sizes_np) > 0 else 0,
        'duration': duration,
        'tcp_connections': len(stats['tcp_connections']),
        'inbound_outbound_ratio': total_inbound / total_outgoing if total_outgoing > 0 else 0,
        'sent_received_ratio': (np.sum(outgoing_lengths_np) / np.sum(inbound_lengths_np)) 
                               if len(inbound_lengths_np) > 0 and np.sum(inbound_lengths_np) > 0 else 0,
        'out_icmp_percentage': calculate_percentage(
            stats['protocol_counts']['out'].get('ICMP', 0), 
            total_outgoing
        ),
        'in_icmp_percentage': calculate_percentage(
            stats['protocol_counts']['in'].get('ICMP', 0), 
            total_inbound
        ),
        'in_len_uniquelen': len(set(stats['inbound_lengths'])) / len(stats['inbound_lengths']) 
                           if stats['inbound_lengths'] else 0,
        'out_mean_flownumpkts': np.mean(list(stats['tcp_connections'].values())) 
                               if stats['tcp_connections'] else 0,
        'out_mean_len': np.mean(outgoing_lengths_np) if len(outgoing_lengths_np) > 0 else 0,
        'in_median_uniquelen': np.median(sorted(set(stats['inbound_lengths']))) if stats['inbound_lengths'] else 0,
        'out_totalpkts': total_outgoing,
        'in_totalpkts': total_inbound,
        'out_percentage': calculate_percentage(total_outgoing, total_packet_count),
        'in_percentage': calculate_percentage(total_inbound, total_packet_count)
    }
