package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"
)

func main() {
	ip := flag.String("a", "", "The IP address of the Chord client")
	port := flag.Int("p", -1, "The port that the Chord client will bind to and listen on")
	runningIp := flag.String("ja", "", "The IP address of the machine running Chord")
	runningPort := flag.Int("jp", -1, "The port number of the machine running Chord")
	stabilizeTime := flag.Int("ts", 30000, "The time in milliseconds between invocations of stabilize")
	fixFingersTime := flag.Int("tff", 10000, "The time in milliseconds between invocations of fix fingres")
	checkPredecessorTime := flag.Int("tcp", 30000, "The time in milliseconds between invocations of check predecessor")
	nSuccessors := flag.Int("r", 1, "The number of successors mainted by the Chord client")
	//identifier := flag.String("i", "", "The identifier (ID) assigned to the Chord client which will override the ID computed by hash function")
	flag.Parse()

	fmt.Printf("ip: %s\n", *ip)
	fmt.Printf("port: %v\n", *port)
	fmt.Printf("runningIp: %s\n", *runningIp)
	fmt.Printf("runningPort: %v\n", *runningPort)
	fmt.Printf("Time between stabilize call: %v\n", *stabilizeTime)
	fmt.Printf("Time between fix finger call: %v\n", *fixFingersTime)
	fmt.Printf("Time between check predecessor call: %v\n", *checkPredecessorTime)
	fmt.Printf("The number of successors: %v\n", *nSuccessors)

	// Depending on command-line arguments, either create a new ring or join an existing one
	node := NewNode(*ip, *port)
	node.server()

	if len(*runningIp) == 0 || *runningPort == -1 {
		node.Create()

		//go func() {
		//	time.Sleep(1 * time.Millisecond)
		//}()

	} else {
		addr := fmt.Sprintf(*runningIp + ":" + strconv.Itoa(*runningPort))
		node.Join(addr)
	}

	// Periodic tasks
	go func() {
		for {
			node.Stabilize()
			time.Sleep(time.Duration(*stabilizeTime) * time.Millisecond)
		}
	}()

	go func() {
		for {
			node.FixFingers()
			time.Sleep(time.Duration(*fixFingersTime) * time.Millisecond)
		}
	}()

	go func() {
		for {
			node.CheckPredecessor()
			time.Sleep(time.Duration(*checkPredecessorTime) * time.Millisecond)
		}
	}()

	go func() {
		for {
			node.CheckSuccessor()
			time.Sleep(time.Duration(*checkPredecessorTime) * time.Millisecond)
		}
	}()

	r := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("> ")
		input, err := r.ReadString('\n')
		if err != nil {
			fmt.Println("Error reading", err)
		}
		tmp := strings.Split(input, " ")
		tmp1 := strings.TrimSpace(tmp[0])
		cmd := strings.ToLower(tmp1)

		if cmd == "lookup" {
			fileName := strings.TrimSpace(tmp[1])
			node.Lookup(fileName)

		} else if cmd == "storefile" {
			fileName := strings.TrimSpace(tmp[1])
			node.StoreFile(fileName)

		} else if cmd == "printstate" {
			node.PrintState()

		} else if cmd == "quit" {
			fmt.Println("Quiting...")
			node.MoveBucket()
			os.Exit(0)
		} else {

			fmt.Println("cmd not supported")
		}

	}

}
