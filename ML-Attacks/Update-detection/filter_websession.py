from scapy.all import rdpcap, wrpcap, DNS, TCP, UDP, IP
import os

def is_valid_192(ip):
    if ip.startswith("192.") and not (ip.endswith(".1") or ip.endswith(".255")):
        return True
    return False

def split_pcap_by_ip(input_pcap, output_dir="split_pcaps"):
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


def is_dns(pkt):
    if not UDP in pkt:
        return False
    return (UDP in pkt and (pkt[UDP].sport == 53 or pkt[UDP].dport == 53)) and DNS in pkt

def is_tcp(pkt):
    return TCP in pkt

def extract_sessions(pcap_file, output_dir="web_sessions"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    packets = rdpcap(pcap_file)
    sessions = []
    current_session = []
    state = "IDLE"
    dns_seen = 0
    session_count = 0

    for pkt in packets:
        if is_dns(pkt):
            if state == "IDLE":
                # Start of a new session
                current_session = [pkt]
                state = "DNS1"
                dns_seen = 1
            elif state == "DNS1":
                current_session.append(pkt)
                dns_seen += 1
            elif state == "TCP1":
                if len([p for p in current_session if TCP in p]) >= 3:
                    session_count += 1
                    filename = os.path.join(output_dir, f"session_{session_count}.pcap")
                    wrpcap(filename, current_session)
                    print(f"Saved {filename} with {len(current_session)} packets")
                else:
                    print("Skipped session with too few TCP packets.")
                # Start new session
                current_session = [pkt]
                state = "DNS1"
                dns_seen = 1
        elif is_tcp(pkt):
            if state == "DNS1" or state == "TCP1":
                current_session.append(pkt)
                state = "TCP1"
        else:
            if state != "IDLE":
                current_session.append(pkt)

    # Save final session if valid
    if current_session and state == "TCP1":
        session_count += 1
        filename = os.path.join(output_dir, f"session_{session_count}.pcap")
        wrpcap(filename, current_session)
        print(f"Saved {filename} with {len(current_session)} packets")

def run_pipeline(input_pcap):
    split_output_dir = "split_pcaps_web"
    session_output_base = "web_sessions"

    split_pcap_by_ip(input_pcap, split_output_dir)

    for filename in os.listdir(split_output_dir):
        if filename.endswith(".pcap"):
            ip = filename.replace(".pcap", "")
            ip_session_dir = os.path.join(session_output_base, ip)
            filepath = os.path.join(split_output_dir, filename)
            extract_sessions(filepath, ip_session_dir)

if __name__ == "__main__":
    run_pipeline("web-traffic.pcap")

