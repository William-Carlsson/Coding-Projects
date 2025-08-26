FROM golang:1.21.3

WORKDIR /app

COPY go.mod ./

# There are no dependencies for this project, so this will do nothing.
RUN go mod download && go mod verify

COPY . ./

RUN CGO_ENABLED=0 GOOS=linux go build -o /myapp

EXPOSE 8080 8081

CMD ["/myapp", "8080"]