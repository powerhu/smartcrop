# brew install protobuf-compiler
# brew install golang-goprotobuf-dev
export GO111MODULE=on
go get github.com/golang/protobuf/protoc-gen-go
go get google.golang.org/grpc/cmd/protoc-gen-go-grpc@v1.0
export PATH="$PATH:$(go env GOPATH)/bin"

protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative facedetection/face_detection.proto
