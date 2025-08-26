import json
import os
import numpy as np
from scapy.all import rdpcap, TCP, IP
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pyshark
import hashlib

def extract_features_from_pcap(file_path, max_len=67, esp32_prefix="192.168"):
    packets = rdpcap(file_path)
    session = []
    last_time = None
    base_time = None

    sni_map = {}
    try:
        pcap = pyshark.FileCapture(file_path)
        for i, py_pkt in enumerate(pcap):
            try:
                if "tls" in py_pkt and hasattr(py_pkt.tls, 'handshake_type') and py_pkt.tls.handshake_type == '1':
                    sni_val = getattr(py_pkt.tls, 'handshake_extensions_server_name', "")
                    sni_map[i] = sni_val
            except Exception:
                continue
        pcap.close()
    except Exception:
        pass

    for i, pkt in enumerate(packets):
        if not pkt.haslayer(IP):
            continue

        if base_time is None:
            base_time = pkt.time

        timestamp = float(pkt.time - base_time)
        interarrival_time = timestamp - last_time if last_time is not None else 0
        last_time = timestamp

        length = len(pkt)
        proto_id = pkt[IP].proto

        src_ip = pkt[IP].src
        direction = 1 if src_ip.startswith(esp32_prefix) else -1

        src_port = pkt.sport if hasattr(pkt, 'sport') else 0
        dst_port = pkt.dport if hasattr(pkt, 'dport') else 0

        tcp_flag = "NONE"
        flag_encoding = -1
        seq = -1
        ack = -1
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            tcp_flag = tcp.sprintf("%TCP.flags%")
            flag_encoding = tcp_flag_encoding(tcp_flag)
            seq = int(tcp.seq)
            ack = int(tcp.ack)

        # SNI and ClientHello tracking
        is_client_hello = 1 if i in sni_map else 0
        sni = sni_map[i] if i in sni_map else None
        sni_id = hash_sni(sni)

        session.append([
            length,
            interarrival_time,
            timestamp,
            direction,
            proto_id,
            is_client_hello,
            src_port,
            dst_port,
            sni_id,
            flag_encoding,
            seq,
            ack
        ])

        if len(session) >= max_len:
            break

    return pad_sequence(session, max_len)


def tcp_flag_encoding(flag_str):
    flags = ['N', 'C', 'E', 'U', 'A', 'P', 'R', 'S', 'F']

    if flag_str in flags:
        index = flags.index(flag_str)
        return int(1 << (len(flags) - 1 - index))
    return 0


def hash_sni(sni_str, max_buckets=1000):
    if not sni_str or sni_str == -1:
        return -1
    return int(hashlib.md5(sni_str.encode()).hexdigest(), 16) % max_buckets


def pad_sequence(seq, max_len=67):
    padded = []
    for i in range(max_len):
        if i < len(seq):
            padded.append(seq[i])
        else:
            # Create a zeroed packet with the same structure
            padded.append([0] * 12)
    return padded

def list_pcap_files(root_folder):
    pcap_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith(".pcap"):
                pcap_files.append(os.path.join(dirpath, f))
    return pcap_files

def remap_labels(feature_matrices):
    sni_values = set()
    flag_values = set()

    # Collect all SNI and flag values
    for mat in feature_matrices:
        for pkt in mat:
            if isinstance(pkt[8], str):
                sni_values.add(pkt[8])
            if isinstance(pkt[9], str):
                flag_values.add(pkt[9])

    sni_map = {val: idx for idx, val in enumerate(sorted(sni_values))}
    flag_map = {val: idx for idx, val in enumerate(sorted(flag_values))}

    # Apply remapping only to values that are strings
    for mat in feature_matrices:
        for pkt in mat:
            if isinstance(pkt[8], str):
                pkt[8] = sni_map.get(pkt[8], -1)
            # Leave -1 as-is
            if isinstance(pkt[9], str):
                pkt[9] = flag_map[pkt[9]]

    return feature_matrices


def extract_all_features_from_multiple_folders(folders, max_workers=4):
    all_files = []
    for folder in folders:
        all_files.extend(list_pcap_files(folder))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(extract_features_from_pcap, all_files), total=len(all_files)))

    return remap_labels(results)

def get_label_from_filename(file_name):
    # Label as OTA if "ota" is in the file name (case insensitive), else non-OTA
    return 1 if 'ota' in file_name.lower() else 0

def extract_features_with_labels_from_folders(folders, max_len=67, max_workers=4):
    all_files = []
    for folder in folders:
        all_files.extend(list_pcap_files(folder))

    labeled_sessions = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for file in tqdm(all_files):
            label = get_label_from_filename(file)
            features = extract_features_from_pcap(file, max_len)
            labeled_sessions.append((features, label))  # Append the label to the features

    return labeled_sessions


def extract_features_with_len(args):
    path, max_len = args
    return extract_features_from_pcap(path, max_len)


def extract_update_version_dataset_from_folder(root_folder, max_len=1900, max_workers=4):
    all_files = list_pcap_files(root_folder)
    
    # Extract version names from filenames
    version_names = [os.path.basename(f).split("_")[0] for f in all_files]
    unique_versions = sorted(set(version_names))
    version_map = {ver: idx for idx, ver in enumerate(unique_versions)}

    # Prepare arguments
    args_list = [(f, max_len) for f in all_files]

    # Extract features in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        features = list(tqdm(executor.map(extract_features_with_len, args_list), total=len(args_list)))

    labels = [version_map[os.path.basename(f).split("_")[0]] for f in all_files]
    labeled_sessions = list(zip(features, labels))

    return labeled_sessions, version_map




if __name__ == "__main__":
    folders = ["D:\MasterThesisData\extracted_ota_sessions", "D:\MasterThesisData\heartbeat_sessions", "D:\MasterThesisData\web_sessions"]
    labeled_feature_matrices = extract_features_with_labels_from_folders(folders, max_workers=8)

    X = [features for features, _ in labeled_feature_matrices]
    y_version = [label for _, label in labeled_feature_matrices]

    
    X = [features for features, _ in labeled_feature_matrices]
    y = [label for _, label in labeled_feature_matrices]

    with open("update_detection_data.json", "w") as f:
        json.dump({"X": X, "y": y}, f)


