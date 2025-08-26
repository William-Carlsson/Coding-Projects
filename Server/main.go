package main

import (
	"fmt"
	"os"
	"strconv"
)

func main() {

	if len(os.Args) != 2 {
		fmt.Println("Usage: go run main.go <port>")
		return
	}

	// Check if valid port
	port, err := strconv.Atoi(os.Args[1])
	if err != nil {
		fmt.Println("Port must be an integer.")
		return
	}

	if port < 0 || port > 65535 {
		fmt.Println("Port must be in [0, 65535]")
		return
	}

	// Create the main server
	mainServer := NewMainServer(port)
	go mainServer.Run()

	// Create the proxy server
	proxyServer := NewProxyServer(port+1, mainServer.server.listenAddr)
	go proxyServer.Run()

	select {}

}
