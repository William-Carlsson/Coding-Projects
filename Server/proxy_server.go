package main

import (
	"io"
	"log"
	"net/http"
)

// ProxyServer represents a proxy server that handles specific HTTP requests.
type ProxyServer struct {
	server         *Server
	mainServerAddr string
}

// NewProxyServer creates a new ProxyServer instance with the specified port.
func NewProxyServer(port int, listenAddr string) *ProxyServer {
	ps := &ProxyServer{mainServerAddr: listenAddr}
	ps.server = NewServer(port, ps.handleRequest)
	return ps
}

// Run starts the ProxyServer and begins handling incoming HTTP requests.
func (ps *ProxyServer) Run() {
	ps.server.Run()
}

// handleRequest is responsible for processing incoming HTTP requests.
func (ps *ProxyServer) handleRequest(res *CustomResponseWriter, req *http.Request) {
	switch req.Method {
	case http.MethodGet:
		ps.handleRequestGET(res, req)
	default:
		HandleMethodNotImplemented(res)
	}
}

// handleRequestGET handles GET requests for the ProxyServer and responds with "Method Not Implemented."
func (ps *ProxyServer) handleRequestGET(res *CustomResponseWriter, req *http.Request) {

	// Create an HTTP client for forwarding requests.
	client := &http.Client{}

	// Clone request and set relevant fields
	reqClone := new(http.Request)
	*reqClone = *req
	reqClone.RequestURI = "" // Required to be unset
	reqClone.URL.Scheme = "http"
	reqClone.URL.Host = ps.mainServerAddr

	// Forward the request to the main server
	resp, err := client.Do(reqClone)
	if err != nil {
		log.Println("[PROXY] Error forwarding request to client:", err)
		http.Error(res, err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Copy the server response headers to proxy's response header
	CopyHeaders(res.Header(), resp.Header)
	res.WriteHeader(resp.StatusCode)

	// Send the server response body to the client
	_, err = io.Copy(res, resp.Body)
	if err != nil {
		log.Println("[PROXY] Error sending server response body to the client:", err)
		http.Error(res, err.Error(), http.StatusBadGateway)
		return
	}

}
