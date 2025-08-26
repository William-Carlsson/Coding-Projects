package main

type FindSuccArgs struct {
	Id    string
	Start Node
}

type FindSuccReply struct {
	Node   string
	Status bool
}

type GetSuccArgs struct {
}

type GetSuccReply struct {
	PredecessorAddress string
}

type NotifySuccArgs struct {
	Node string
}

type NotifySuccReply struct {
}

type PingArgs struct {
}

type PingReply struct {
	Status string
}

type SendFileArgs struct {
	Filename string
}

type SendFileReply struct {
}

type GetIdArgs struct {
}

type GetIdReply struct {
	Id string
}

type CheckFileArgs struct {
	Key string
}

type CheckFileReply struct {
	Owner bool
}

type MoveBucketArgs struct {
	Bucket  map[string]string
	Address string
}

type MoveBucketReply struct {
}
