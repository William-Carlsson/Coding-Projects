import pyshark
import os
from scapy.all import rdpcap, wrpcap
from tqdm import tqdm

def contains_n_client_hello(pcap_file, target_hello_count=3):
    cap = pyshark.FileCapture(pcap_file, keep_packets=False)
    hello_indices = []

    for i, pkt in enumerate(cap):
        try:
            if "tls" in pkt and hasattr(pkt.tls, 'handshake_type'):
                if pkt.tls.handshake_type == '1':  # Client Hello
                    hello_indices.append(i)
                    if len(hello_indices) == target_hello_count:
                        break
        except AttributeError:
            continue
        except Exception:
            continue

    cap.close()
    if len(hello_indices) < target_hello_count:
        return None
    return hello_indices[-1]

def extract_fixed_window_after_3_client_hellos(input_dir, output_dir="extracted_ota_sessions", max_files=28800, fixed_window=67):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = [f for f in os.listdir(input_dir) if f.endswith(".pcap")]
    saved = 0

    for filename in tqdm(all_files, desc="Extracting OTA patterns"):
        if saved >= max_files:
            break

        full_path = os.path.join(input_dir, filename)
        third_hello_index = contains_n_client_hello(full_path)

        if third_hello_index is None:
            continue

        raw_packets = rdpcap(full_path)
        if len(raw_packets) < fixed_window:
            continue  # Skip if file is too short

        sliced_packets = raw_packets[:fixed_window]
        save_path = os.path.join(output_dir, f"ota_{saved+1}.pcap")
        wrpcap(save_path, sliced_packets)
        saved += 1

    print(f"Extracted {saved} OTA update sequences (first {fixed_window} packets) to: {output_dir}")

if __name__ == "__main__":
    extract_fixed_window_after_3_client_hellos(
        input_dir="Traffic", 
        output_dir="extracted_ota_sessions", 
        fixed_window=67
    )
