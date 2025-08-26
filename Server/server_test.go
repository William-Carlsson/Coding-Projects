package main

import (
	"bytes"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"testing"
)

type testCase struct {
	name, method string
	expected     expectedValue
	url          string
	fileName     string
}

type expectedValue struct {
	statusCode int
	body       string
}

func TestHandlerGET(t *testing.T) {

	testCases := []testCase{
		{name: "Testing GET method with Invaild file extension (htmll)",
			method:   http.MethodGet,
			url:      "http://localhost:8080/indexo.htmll",
			expected: expectedValue{statusCode: http.StatusBadRequest, body: ErrBadRequestMsg}},
		{name: "Testing GET method with vaild file extension (html) and a non existing file",
			method:   http.MethodGet,
			url:      "http://localhost:8080/indexo.html",
			expected: expectedValue{statusCode: http.StatusNotFound, body: ErrNotFoundMsg}},
		{name: "Testing GET method with vaild file extension (txt) and an existing file",
			method:   http.MethodGet,
			url:      "http://localhost:8080/textfile.txt",
			expected: expectedValue{statusCode: http.StatusOK, body: readFile("textfile.txt")}},
		{name: "Testing GET method with vaild file extension (html) and an existing file",
			method:   http.MethodGet,
			url:      "http://localhost:8080/index.html",
			expected: expectedValue{statusCode: http.StatusOK, body: readFile("index.html")}},
		{name: "Testing GET method with vaild file extension (css) and an existing file",
			method:   http.MethodGet,
			url:      "http://localhost:8080/style.css",
			expected: expectedValue{statusCode: http.StatusOK, body: readFile("style.css")}},
		{name: "Testing GET method with vaild file extension (jpg) and an existing file",
			method:   http.MethodGet,
			url:      "http://localhost:8080/cat.jpg",
			expected: expectedValue{statusCode: http.StatusOK, body: readFile("cat.jpg")}},
		{name: "Testing GET method with vaild file extension (jpeg) and an existing file",
			method:   http.MethodGet,
			url:      "http://localhost:8080/cat.jpeg",
			expected: expectedValue{statusCode: http.StatusOK, body: readFile("cat.jpeg")}},
		{name: "Testing GET method with vaild file extension (txt) and an existing file (proxy server)",
			method:   http.MethodGet,
			url:      "http://localhost:8081/textfile.txt",
			expected: expectedValue{statusCode: http.StatusOK, body: readFile("textfile.txt")}},
	}

	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}

}

func TestHandlerPOST(t *testing.T) {

	testCases := []testCase{
		{name: "Testing post method (proxy server)",
			method:   http.MethodPost,
			url:      "http://localhost:8081",
			expected: expectedValue{statusCode: http.StatusNotImplemented, body: ErrNotAllowedMsg},
			fileName: "upload_me.txtt"},
		{name: "Testing post method with invaild file extension",
			method:   http.MethodPost,
			url:      "http://localhost:8080",
			expected: expectedValue{statusCode: http.StatusBadRequest, body: ErrBadRequestMsg},
			fileName: "upload_me.txtttt"},
		{name: "Testing post method with vaild file extension",
			method:   http.MethodPost,
			url:      "http://localhost:8080",
			expected: expectedValue{statusCode: http.StatusOK, body: "Created: New file 'upload_me.txt' uploaded"},
			fileName: "upload_me.txt"},
	}

	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}

}

func TestHandlerPUT(t *testing.T) {
	testCases := casesForOtherMethod("PUT")
	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}
}

func TestHandlerDELETE(t *testing.T) {
	testCases := casesForOtherMethod("DELETE")
	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}
}

func TestHandlerPATCH(t *testing.T) {
	testCases := casesForOtherMethod("PATCH")
	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}
}

func TestHandlerCONNECT(t *testing.T) {
	testCases := casesForOtherMethod("CONNECT")
	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}
}

func TestHandlerTRACE(t *testing.T) {
	testCases := casesForOtherMethod("TRACE")
	t.Parallel()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeTestCase(t, tc)
		})
	}
}

func executeTestCase(t *testing.T, tc testCase) {
	client := &http.Client{}
	request := createNewRequest(tc)
	response, err := client.Do(request)
	if err != nil {
		log.Fatal(err)
	}

	responseBody, err := io.ReadAll(response.Body)
	defer response.Body.Close()
	if err != nil {
		log.Fatal(err)
	}

	if response.StatusCode != tc.expected.statusCode {
		t.Errorf("Want status '%d', got '%d'", tc.expected.statusCode, response.StatusCode)
	}

	if string(responseBody) != tc.expected.body {
		t.Errorf("got %q, want %q", string(responseBody), tc.expected)
	}

	if tc.method == "POST" && response.StatusCode == http.StatusOK { // A check for the post request when it was successful
		destPath := fmt.Sprintf("./uploaded/%s", tc.fileName)
		_, err := os.Stat(destPath)
		if os.IsNotExist(err) {
			t.Errorf("There was no file created")
		}

		f1, err := os.ReadFile(tc.fileName)
		if err != nil {
			log.Fatal(err)
		}

		f2, err := os.ReadFile(destPath)
		if err != nil {
			log.Fatal(err)
		}

		if !bytes.Equal(f1, f2) {
			t.Errorf("The content of the created file is wrong")
		}
	}
}

func createNewRequest(tc testCase) *http.Request {
	switch tc.method {
	case "POST":
		var crt string
		fileDir, err := os.Getwd()
		fileName := tc.fileName
		filePath := path.Join(fileDir, fileName)
		if err != nil {
			log.Fatal(err)
		}

		file, err := os.Open(filePath)
		if err != nil {
			crt = filepath.Base(filePath)
		} else {
			crt = filepath.Base(file.Name())
		}
		defer file.Close()

		body := &bytes.Buffer{}
		writer := multipart.NewWriter(body)
		part, err := writer.CreateFormFile("file", crt)
		if err != nil {
			log.Fatal(err)
		}
		io.Copy(part, file)
		writer.Close()

		request, err := http.NewRequest("POST", tc.url, body)
		if err != nil {
			log.Fatal(err)
		}
		request.Header.Add("Content-Type", writer.FormDataContentType())
		return request
	default:
		request, err := http.NewRequest(tc.method, tc.url, nil)
		if err != nil {
			log.Fatal(err)
		}
		return request
	}
}

func casesForOtherMethod(method string) []testCase {
	return []testCase{
		{
			name:     fmt.Sprintf("Testing %s method (proxy server)", method),
			method:   method,
			expected: expectedValue{statusCode: http.StatusNotImplemented, body: "Not Allowed: HTTP method not allowed.\n"},
			url:      "http://localhost:8081",
		},
		{
			name:     fmt.Sprintf("Testing %s method", method),
			method:   method,
			expected: expectedValue{statusCode: http.StatusNotImplemented, body: "Not Allowed: HTTP method not allowed.\n"},
			url:      "http://localhost:8080",
		},
	}
}

// readFile reads the file from the specified path and returns a string of the content of that file
func readFile(path string) string {
	filePath := fmt.Sprintf("./uploaded/%s", path)
	dat, err := os.ReadFile(filePath)
	if err != nil {
		log.Fatal(err)
	}
	return string(dat)
}
