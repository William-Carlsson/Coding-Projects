package main

import (
	"fmt"
	"net/http"
	"os"
)

// MainServer represents the main server that handles HTTP requests.
type MainServer struct {
	server *Server
}

// NewMainServer creates a new MainServer instance with the specified port.
func NewMainServer(port int) *MainServer {
	ms := &MainServer{}
	ms.server = NewServer(port, ms.handleRequest)
	return ms
}

// Run starts the MainServer and begins handling incoming HTTP requests.
func (ms *MainServer) Run() {
	ms.server.Run()
}

// handleRequest is responsible for processing incoming HTTP requests.
func (ms *MainServer) handleRequest(res *CustomResponseWriter, req *http.Request) {
	switch req.Method {
	case http.MethodGet:
		ms.handleRequestGET(res, req)
	case http.MethodPost:
		ms.handleRequestPOST(res, req)
	default:
		HandleMethodNotImplemented(res)
	}
}

// handleRequestGET handles GET requests and serves requested files.
func (ms *MainServer) handleRequestGET(res *CustomResponseWriter, req *http.Request) {

	path := req.URL.Path
	if path == "/" {
		// Not referencing a file!
		http.Error(res, ErrBadRequestMsg, http.StatusNotFound)
		return
	}

	filePath := fmt.Sprintf("./uploaded%s", req.URL.Path)

	fileExtension, isValidExt := IsValidFileExtension(filePath)
	if !isValidExt {
		http.Error(res, ErrBadRequestMsg, http.StatusBadRequest)
		return
	}

	file, err := os.Open(filePath)
	if err != nil {
		http.Error(res, ErrNotFoundMsg, http.StatusNotFound)
		return
	}
	defer file.Close()

	fileInfo, err := file.Stat()
	if err != nil {
		http.Error(res, ErrInternalServerErrorMsg, http.StatusInternalServerError)
		return
	}

	header := res.Header()
	header.Set("Content-Type", GetContentType(fileExtension))
	header.Set("Content-Length", fmt.Sprint(fileInfo.Size()))

	res.WriteHeader(http.StatusOK)
	res.MustWriteFile(file)

}

// handleRequestPOST handles POST requests for uploading files.
func (ms *MainServer) handleRequestPOST(res *CustomResponseWriter, req *http.Request) {

	uploadedFile, handler, err := req.FormFile("file")
	if err != nil {
		http.Error(res, ErrBadRequestMsg, http.StatusBadRequest)
		return
	}
	defer uploadedFile.Close()

	savePath := fmt.Sprintf("./uploaded/%s", handler.Filename)

	if _, isValidExt := IsValidFileExtension(savePath); !isValidExt {
		http.Error(res, ErrBadRequestMsg, http.StatusBadRequest)
		return
	}

	_, err = SaveFile(savePath, uploadedFile)
	if err != nil {
		http.Error(res, ErrInternalServerErrorMsg, http.StatusInternalServerError)
	}

	res.WriteHeader(http.StatusOK)
	res.Write([]byte(fmt.Sprintf("Created: New file '%s' uploaded", handler.Filename)))

}
