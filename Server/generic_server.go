package main

import (
	"bufio"
	"log"
	"net"
	"net/http"
	"strconv"
	"time"
)

type Server struct {
	listenAddr           string
	port                 int
	ln                   net.Listener
	maxConn              int
	activeConn           int
	done                 chan struct{}
	handleConnectionFunc func(w *CustomResponseWriter, r *http.Request)
}

func NewServer(port int, handleConnectionFunc func(w *CustomResponseWriter, r *http.Request)) *Server {
	server := &Server{
		listenAddr:           ":" + strconv.Itoa(port),
		port:                 port,
		maxConn:              10,
		activeConn:           0,
		done:                 make(chan struct{}),
		handleConnectionFunc: handleConnectionFunc,
	}

	listener, err := net.Listen("tcp", server.listenAddr)
	if err != nil {
		log.Fatal("Failed creating new server:", err)
		return nil
	}
	server.ln = listener

	return server
}

func (s *Server) Run() {

	log.Println("Server is running on", s.listenAddr)

	timeDelay := 1 * time.Second

	for {
		select {
		case <-s.done:
		// TODO: Gracefully shut down server
		default:
			// Server can only serve a limited number connections at once
			if s.activeConn > s.maxConn {
				time.Sleep(timeDelay)
				continue
			}

			// Accept a new connection
			conn, err := s.ln.Accept()
			if err != nil {
				log.Println("Error accepting new connection:", err)
				continue
			}

			s.activeConn++
			log.Printf("[%s] Accepted new connection %s\n", s.listenAddr, conn.RemoteAddr())

			w := NewCustomResponseWriter(conn)

			request, err := readRequest(conn)
			if err != nil {
				log.Printf("[%s] Error parsing incoming request from %s: %s\n", s.listenAddr, conn.RemoteAddr(), err)
				http.Error(w, ErrBadRequestMsg, http.StatusBadRequest)
				conn.Close()
				s.activeConn--
				continue
			}

			go func() {
				defer conn.Close()
				s.handleConnectionFunc(w, request)
				s.activeConn--
			}()

		}
	}

}

func readRequest(conn net.Conn) (r *http.Request, e error) {
	reader := bufio.NewReader(conn)
	request, err := http.ReadRequest(reader)
	return request, err
}
