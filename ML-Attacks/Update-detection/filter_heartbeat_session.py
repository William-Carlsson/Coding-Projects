import pyshark
import os
from scapy.all import rdpcap, wrpcap, IP

def is_valid_192(ip):
    if ip.startswith("192.") and not (ip.endswith(".1") or ip.endswith(".255")):
        return True
    return False

def split_pcap_by_ip(input_pcap, output_dir="split_pcaps_heartbeat"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    packets = rdpcap(input_pcap)
    ip_to_packets = {}

    for pkt in packets:
        if IP in pkt:
            src = pkt[IP].src
            dst = pkt[IP].dst
            for ip in (src, dst):
                if is_valid_192(ip):
                    ip_to_packets.setdefault(ip, []).append(pkt)

    for ip, pkts in ip_to_packets.items():
        filename = os.path.join(output_dir, f"{ip.replace('.', '_')}.pcap")
        wrpcap(filename, pkts)
        print(f"Wrote {len(pkts)} packets to {filename}")


def extract_heartbeat_sessions(pcap_file, output_dir="heartbeat_sessions", ack_seq_target="1639",ip_subdir_name=None):

    if ip_subdir_name:
        output_dir = os.path.join(output_dir, ip_subdir_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    cap = pyshark.FileCapture(pcap_file, keep_packets=False)
  
    current_session = []
    tracking = False

    # Load raw packets with scapy to extract and save later
    raw_packets = rdpcap(pcap_file)


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    session_count = 0

    for i, pkt in enumerate(cap):
        try:
            if "tcp" not in pkt:
                continue

            seq_num = pkt.tcp.seq
            
            if not tracking:
                if pkt.tcp.flags == "0x0002" and seq_num == "0":  # SYN and seq==0
                    tracking = True
                    current_session = [i]
            else:
                current_session.append(i)
                if pkt.tcp.flags == "0x0010" and seq_num == ack_seq_target:  # ACK and expected ack seq
                    extracted = [raw_packets[j] for j in current_session]
                    session_count += 1
                    fname = f"{output_dir}/heartbeat_{session_count}.pcap"
                    wrpcap(fname, extracted)
                    print(f"Saved session to {fname}")
                    tracking = False
                    current_session = []
        except AttributeError:
            continue

    cap.close()

def run_pipeline(input_pcap):
    split_dir = "split_pcaps_heartbeat"
    session_dir = "heartbeat_sessions"

    print(f"Splitting pcap by IPs in {input_pcap}...")
    split_pcap_by_ip(input_pcap, output_dir=split_dir)

    print("Extracting heartbeat sessions from split pcaps...")
    for filename in os.listdir(split_dir):
        if filename.endswith(".pcap"):
            split_pcap_path = os.path.join(split_dir, filename)
            ip_subdir_name = filename.replace(".pcap", "")  # e.g., "192_168_137_45"
            print(f"Processing {split_pcap_path} into {ip_subdir_name}/...")
            extract_heartbeat_sessions(
                pcap_file=split_pcap_path,
                output_dir=session_dir,
                ip_subdir_name=ip_subdir_name
            )

if __name__ == "__main__":
    run_pipeline("heartbeat-traffic.pcap")
