package main

import (
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

// SaveFile saves an uploaded file to the specified path and returns the number of bytes written.
func SaveFile(savePath string, uploadedFile multipart.File) (int64, error) {

	newFile, err := os.OpenFile(savePath, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		return 0, err
	}
	defer newFile.Close()

	n, err := io.Copy(newFile, uploadedFile)
	if err != nil {
		return 0, err
	}

	return n, nil

}

// IsValidFileExtension checks if the file extension is valid and returns the file extension and a boolean indicating validity.
func IsValidFileExtension(filePath string) (string, bool) {
	fileExtension := filepath.Ext(filePath)
	validFileExtensions := map[string]bool{
		".html": true,
		".txt":  true,
		".gif":  true,
		".jpeg": true,
		".jpg":  true,
		".css":  true,
	}
	return fileExtension, validFileExtensions[fileExtension]
}

// GetContentType returns the content type associated with the provided file extension.
func GetContentType(fileExtension string) string {
	return map[string]string{
		".html": "text/html",
		".txt":  "text/plain",
		".gif":  "image/gif",
		".jpeg": "image/jpeg",
		".jpg":  "image/jpeg",
		".css":  "text/css",
	}[fileExtension]
}

func CopyHeaders(dst, src http.Header) {
	for key, values := range src {
		for _, value := range values {
			dst.Add(key, value)
		}
	}
}

const (
	ErrBadRequestMsg          = "Bad Request: <msg>"
	ErrNotFoundMsg            = "Not Found: <msg>"
	ErrInternalServerErrorMsg = "Internal Server Error: Something unexpected happened in the server"
	ErrNotAllowedMsg          = "Not Allowed: HTTP method not allowed."
)

// HandleMethodNotImplemented sends a "Not Implemented" HTTP response with the provided CustomResponseWriter.
func HandleMethodNotImplemented(res *CustomResponseWriter) {
	http.Error(res, ErrNotAllowedMsg, http.StatusNotImplemented)
}
