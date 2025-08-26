package main

import (
	"fmt"
	"log"
	"math/big"
	"net"
	"net/http"
	"net/rpc"
)

type Node struct {
	ID          *big.Int
	IP          string
	Port        int
	Address     string
	Finger      []string
	Successor   string
	Predecessor string
	Bucket      map[string]string
}

func NewNode(ip string, port int) *Node {
	address := fmt.Sprintf(ip+":%v", port)
	id := hashString(address)

	node := &Node{
		ID:          id,
		IP:          ip,
		Port:        port,
		Address:     address,
		Finger:      make([]string, 161), // Assuming SHA1 160-bit
		Predecessor: "",
		Bucket:      make(map[string]string),
	}

	node.Successor = node.Address

	return node
}

func (n *Node) GetId(args *GetIdArgs, reply *GetIdReply) error {
	reply.Id = n.ID.String()
	return nil
}

func (n *Node) GetPredecessor(args *GetSuccArgs, reply *GetSuccReply) error {
	reply.PredecessorAddress = n.Predecessor
	return nil
}

func (n *Node) Create() {
	n.Successor = n.Address
	n.Finger[0] = n.Address
	n.Predecessor = ""
}

func (n *Node) Join(nodeAddr string) {
	n.Predecessor = ""
	n.Successor = n.Find(n.ID.String(), nodeAddr)
	n.PrintState()
}

func (n *Node) Stabilize() {

	fmt.Println("Stabilizing!")

	succ := n.Successor

	predec := ""
	if succ != n.Address {
		args := GetSuccArgs{}
		reply := GetSuccReply{}
		err := call(succ, "Node.GetPredecessor", &args, &reply)
		if err != nil {
			log.Println("[Stabilize] Failed to contact successor for their predecessor")
			//log.Println("succ inside Stabilize: ", succ)
			//log.Println(err)
		}
		predec = reply.PredecessorAddress
	} else {
		predec = n.Predecessor
	}

	if predec != "" && between(n.ID, hashString(predec), hashString(succ), false) {
		n.Successor = predec
		a := NotifySuccArgs{Node: predec}
		r := NotifySuccReply{}
		err := call(succ, "Node.Notify", &a, &r)
		if err != nil {
			log.Println("[Stabilize] Failed to notify previous successor about topology change")
			//log.Println("n.Successor in stabilize when notifying in between: ", n.Successor)
			//log.Println("[1] RPC to Node.Notify failed: ", err)
		}
	}

	if n.Successor == n.Address {
		return
	}

	a := NotifySuccArgs{Node: n.Address}
	r := NotifySuccReply{}
	err2 := call(n.Successor, "Node.Notify", &a, &r)
	if err2 != nil {
		log.Println("Failed to contact my successor")
		//log.Println("[2] RPC to Node.Notify failed: ", err2)
		//log.Println("n.Successor in stabilize when notifying at the end: ", n.Successor)
	}
}

func (n *Node) Notify(args *NotifySuccArgs, reply *NotifySuccReply) error {
	nodeAddr := args.Node
	if n.Predecessor == "" || between(hashString(n.Predecessor), hashString(nodeAddr), n.ID, false) {
		n.Predecessor = nodeAddr
	}
	return nil
}

func (n *Node) FixFingers() {

	for i := 1; i < len(n.Finger); i++ {

		// n + 2^(i-1) mod 2^m
		targetID := jump(n.Address, i)

		// find successor of node with id=targetID
		succ := n.Find(targetID.String(), n.Successor)
		if succ == "" {
			n.Finger[i] = ""
			continue
		}

		n.Finger[i] = succ
	}
}

func (n *Node) FindSuccessor(args *FindSuccArgs, reply *FindSuccReply) error {

	id := new(big.Int)
	id.SetString(args.Id, 10)

	if between(n.ID, id, hashString(n.Successor), true) {
		reply.Status = true
		reply.Node = n.Successor
		return nil
	}

	reply.Status = false
	reply.Node = n.closestPrecedingNode(id)
	return nil
}

func (n *Node) closestPrecedingNode(id *big.Int) string {
	for i := len(n.Finger) - 1; i >= 1; i-- {
		//if n.Finger[i] == "" {
		//	continue
		//}
		if between(n.ID, hashString(n.Finger[i]), id, true) {
			return n.Finger[i]
		}
	}
	return n.Successor
}

func (n *Node) Find(id string, startNode string) string {

	if startNode == "" {
		return ""
	}

	tmp := new(big.Int)
	tmp.SetString(id, 10)

	found := false
	nextNodeAddr := startNode

	i := 0
	for !found && i < 32 {
		arg := FindSuccArgs{Id: tmp.String()}
		rep := FindSuccReply{}
		err := call(nextNodeAddr, "Node.FindSuccessor", &arg, &rep)
		if err != nil {
			log.Println("[Find] Failed to contact next node in ring during find process")
			//log.Println("nextNodeAddr inside find: ", nextNodeAddr)
			//log.Println(err)
		}
		found, nextNodeAddr = rep.Status, rep.Node
		i += 1
	}

	return nextNodeAddr
}	

func (n *Node) CheckPredecessor() {

	if n.Predecessor == "" {
		return
	}

	a := PingArgs{}
	r := PingReply{}
	err := call(n.Predecessor, "Node.Ping", &a, &r)
	if err != nil {
		log.Println("[CheckPredecessor] Pinging predecessor failed. Resetting it to none.")
		// Could not dial, assume node is dead
		// Empty predecessor, it'll get fixed during FixFingers call
		n.Predecessor = ""
	}
}

func (n *Node) CheckSuccessor() {

	if n.Successor == "" {
		return
	}

	a := PingArgs{}
	r := PingReply{}
	err := call(n.Successor, "Node.Ping", &a, &r)
	if err != nil {
		fmt.Println("[CheckSuccessor] Pinging successor failed. Resetting it to ourself.")
		// Could not dial, assume node is dead
		// We set successor to ourself
		// Stabilize will automatically fix this whenever it's called next
		n.Successor = n.Address
	}
}

func (n *Node) Ping(args *PingArgs, reply *PingReply) error {
	reply.Status = "I am here!"
	return nil
}

func (n *Node) StoreFile(filename string) {

	key := hashString(filename)
	succ := n.Find(key.String(), n.Successor)

	// If this node is the correct node to store the file
	if succ == n.Address {
		n.Bucket[key.String()] = filename
		fmt.Println("[StoreFile] File stored successfully.")
	} else {
		// Send file to the motherfuckeruv
		args := SendFileArgs{Filename: filename}
		reply := SendFileReply{}
		err := call(succ, "Node.ReceiveFile", &args, &reply)
		if err != nil {
			log.Println("[StoreFile] Failed to contact successor")
			//log.Println("succ inside StoreFile: ", succ)
		}
	}

	fmt.Println("[StoreFile] Successfully stored file on node with address: ", succ)
}

func (n *Node) ReceiveFile(args *SendFileArgs, reply *SendFileReply) error {
	key := hashString(args.Filename)
	n.Bucket[key.String()] = args.Filename
	return nil
}

func (n *Node) ReceiveBucket(args *MoveBucketArgs, reply *MoveBucketReply) error {
	for key, val := range args.Bucket {
		n.Bucket[key] = val
	}
	return nil
}

func (n *Node) MoveBucket() {
	arg := MoveBucketArgs{Bucket: n.Bucket, Address: n.Address}
	reply := MoveBucketReply{}
	err := call(n.Successor, "Node.ReceiveBucket", &arg, &reply)
	if err != nil {
		log.Println("[MoveBucker] Failed to contact successor to the bucket")
		//log.Printf("Move bucket has failed: %s\n", error.Error(err))
	}
}

func (n *Node) Lookup(fileName string) {

	key := hashString(fileName)
	succ := n.Find(key.String(), n.Successor)

	a := CheckFileArgs{Key: key.String()}
	r := CheckFileReply{}
	err := call(succ, "Node.IfOwner", &a, &r)
	if err != nil {
		log.Println("[Lookup] Failed to contact owner of key")
		// log.Println("succ inside Lookup failed: ", err)
	}

	if r.Owner {
		// r.Node is the address of the node. I know, naming is amazing!
		fmt.Println("[Lookup] Node Identifier: ", hashString(succ))
		fmt.Println("[Lookup] Node Address: ", succ)
	} else {
		fmt.Println("[Lookup] No owner for file: ", fileName)
	}

}

func (n *Node) IfOwner(args *CheckFileArgs, reply *CheckFileReply) error {
	_, ok := n.Bucket[args.Key]
	reply.Owner = ok
	return nil
}

func (n *Node) PrintState() {

	fmt.Println("\n\n")

	a := GetIdArgs{}
	r := GetIdReply{}
	err := call(n.Successor, "Node.GetId", &a, &r)
	if err != nil {
		log.Println("[PrintState] Failed to contact successor during information fetch")
		//log.Println("n.Successor inside PrintState: ", n.Successor)
		//log.Println(err)
	}

	fmt.Println("##################################################")
	fmt.Println("Printing state for node: ", n.ID)
	fmt.Println("##################################################")
	fmt.Println("[SELF] ID: ", n.ID)
	fmt.Println("[SELF] IP: ", n.IP)
	fmt.Println("[SELF] port: ", n.Port)
	fmt.Println("[SELF] address: ", n.Address)
	fmt.Println("[SUCCESSOR] ID: ", r.Id)
	fmt.Println("[SUCCESSOR] address:port: ", n.Successor)
	fmt.Println("[PREDECESSOR] address:port: ", n.Predecessor)

	// Print finger table information
	for i, finger := range n.Finger {
		if finger != "" {
			a := GetIdArgs{}
			r := GetIdReply{}
			err := call(finger, "Node.GetId", &a, &r)
			if err != nil {
				log.Println("finger at GeTid in printstate: ", finger)
				log.Println(err)
			}
			fmt.Println("#############")
			fmt.Println("Finger ", i)
			fmt.Println("Finger ID: ", r.Id)
			fmt.Println("Finger address:port: ", finger)
		}
	}

	fmt.Println("##################################################")

}

// start a thread that listens for RPCs from other nodes
func (n *Node) server() {
	rpc.Register(n)
	rpc.HandleHTTP()
	address := fmt.Sprintf(n.IP+":%v", n.Port)
	l, e := net.Listen("tcp", address)
	if e != nil {
		log.Fatal("listen error:", e)
	}
	log.Println("Server listening on: ", address)
	go http.Serve(l, nil)
}
