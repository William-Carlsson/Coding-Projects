package main

import (
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
)

// A CustomResponseWrite implements the http.ResponseWriter interface
// and has extended functionality to work with external connections of
// type 'net.Conn'.
type CustomResponseWriter struct {
	header http.Header
	conn   net.Conn
}

// NewCustomResponseWriter creates a new CustomResponseWriter instance
// with the provided net.Conn connection and initializes the header.
func NewCustomResponseWriter(conn net.Conn) *CustomResponseWriter {
	return &CustomResponseWriter{
		header: make(http.Header),
		conn:   conn,
	}
}

// Header returns the response header of the CustomResponseWriter.
func (w *CustomResponseWriter) Header() http.Header {
	return w.header
}

// Write writes the given byte slice to the underlying connection.
// It returns the number of bytes written and any error encountered.
func (w *CustomResponseWriter) Write(bs []byte) (n int, err error) {
	n, err = w.conn.Write(bs)
	return n, err
}

// WriteHeader writes the HTTP response status line, and sends the response header to the underlying connection.
func (w *CustomResponseWriter) WriteHeader(statusCode int) {
	// Write status line
	statusLine := fmt.Sprintf("HTTP/1.1 %d %s\r\n", statusCode, http.StatusText(statusCode))
	w.Write([]byte(statusLine))

	// Write header
	w.header.Set("Connection", "close")
	w.header.Write(w.conn)

	// Don't forget header(s)-body separator!
	w.Write([]byte("\r\n"))
}

// MustWriteFile writes the content of the provided os.File to the underlying connection.
// If an error occurs during the write, it will panic.
func (w *CustomResponseWriter) MustWriteFile(file *os.File) {
	_, err := io.Copy(w.conn, file)
	if err != nil {
		panic(err)
	}
}
