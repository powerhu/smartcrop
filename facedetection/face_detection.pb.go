// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.25.0
// 	protoc        v3.13.0
// source: facedetection/face_detection.proto

package facedetection

import (
	proto "github.com/golang/protobuf/proto"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// This is a compile-time assertion that a sufficiently up-to-date version
// of the legacy proto package is being used.
const _ = proto.ProtoPackageIsVersion4

type DetectedObj struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Lx    int32   `protobuf:"varint,2,opt,name=lx,proto3" json:"lx,omitempty"`        // left top x
	Ly    int32   `protobuf:"varint,3,opt,name=ly,proto3" json:"ly,omitempty"`        // left top y
	Rx    int32   `protobuf:"varint,4,opt,name=rx,proto3" json:"rx,omitempty"`        // right bottom x
	Ry    int32   `protobuf:"varint,5,opt,name=ry,proto3" json:"ry,omitempty"`        // right bottom y
	Score float32 `protobuf:"fixed32,7,opt,name=score,proto3" json:"score,omitempty"` // confidence of object type
}

func (x *DetectedObj) Reset() {
	*x = DetectedObj{}
	if protoimpl.UnsafeEnabled {
		mi := &file_facedetection_face_detection_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DetectedObj) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DetectedObj) ProtoMessage() {}

func (x *DetectedObj) ProtoReflect() protoreflect.Message {
	mi := &file_facedetection_face_detection_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DetectedObj.ProtoReflect.Descriptor instead.
func (*DetectedObj) Descriptor() ([]byte, []int) {
	return file_facedetection_face_detection_proto_rawDescGZIP(), []int{0}
}

func (x *DetectedObj) GetLx() int32 {
	if x != nil {
		return x.Lx
	}
	return 0
}

func (x *DetectedObj) GetLy() int32 {
	if x != nil {
		return x.Ly
	}
	return 0
}

func (x *DetectedObj) GetRx() int32 {
	if x != nil {
		return x.Rx
	}
	return 0
}

func (x *DetectedObj) GetRy() int32 {
	if x != nil {
		return x.Ry
	}
	return 0
}

func (x *DetectedObj) GetScore() float32 {
	if x != nil {
		return x.Score
	}
	return 0
}

type FaceDetRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	ImageData  []byte  `protobuf:"bytes,1,opt,name=imageData,proto3" json:"imageData,omitempty"`     // data raw image data in binary
	Type       string  `protobuf:"bytes,2,opt,name=type,proto3" json:"type,omitempty"`               // image type information, could be jpeg, png, jpg
	ConfThresh float32 `protobuf:"fixed32,3,opt,name=confThresh,proto3" json:"confThresh,omitempty"` // thresh hold to filter low confidence box
}

func (x *FaceDetRequest) Reset() {
	*x = FaceDetRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_facedetection_face_detection_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *FaceDetRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*FaceDetRequest) ProtoMessage() {}

func (x *FaceDetRequest) ProtoReflect() protoreflect.Message {
	mi := &file_facedetection_face_detection_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use FaceDetRequest.ProtoReflect.Descriptor instead.
func (*FaceDetRequest) Descriptor() ([]byte, []int) {
	return file_facedetection_face_detection_proto_rawDescGZIP(), []int{1}
}

func (x *FaceDetRequest) GetImageData() []byte {
	if x != nil {
		return x.ImageData
	}
	return nil
}

func (x *FaceDetRequest) GetType() string {
	if x != nil {
		return x.Type
	}
	return ""
}

func (x *FaceDetRequest) GetConfThresh() float32 {
	if x != nil {
		return x.ConfThresh
	}
	return 0
}

type FaceDetResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	DetObjs []*DetectedObj `protobuf:"bytes,1,rep,name=detObjs,proto3" json:"detObjs,omitempty"` // a list of detected object basic information
}

func (x *FaceDetResponse) Reset() {
	*x = FaceDetResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_facedetection_face_detection_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *FaceDetResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*FaceDetResponse) ProtoMessage() {}

func (x *FaceDetResponse) ProtoReflect() protoreflect.Message {
	mi := &file_facedetection_face_detection_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use FaceDetResponse.ProtoReflect.Descriptor instead.
func (*FaceDetResponse) Descriptor() ([]byte, []int) {
	return file_facedetection_face_detection_proto_rawDescGZIP(), []int{2}
}

func (x *FaceDetResponse) GetDetObjs() []*DetectedObj {
	if x != nil {
		return x.DetObjs
	}
	return nil
}

var File_facedetection_face_detection_proto protoreflect.FileDescriptor

var file_facedetection_face_detection_proto_rawDesc = []byte{
	0x0a, 0x22, 0x66, 0x61, 0x63, 0x65, 0x64, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x2f,
	0x66, 0x61, 0x63, 0x65, 0x5f, 0x64, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x2e, 0x70,
	0x72, 0x6f, 0x74, 0x6f, 0x12, 0x0d, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74,
	0x69, 0x6f, 0x6e, 0x22, 0x63, 0x0a, 0x0b, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74, 0x65, 0x64, 0x4f,
	0x62, 0x6a, 0x12, 0x0e, 0x0a, 0x02, 0x6c, 0x78, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x02,
	0x6c, 0x78, 0x12, 0x0e, 0x0a, 0x02, 0x6c, 0x79, 0x18, 0x03, 0x20, 0x01, 0x28, 0x05, 0x52, 0x02,
	0x6c, 0x79, 0x12, 0x0e, 0x0a, 0x02, 0x72, 0x78, 0x18, 0x04, 0x20, 0x01, 0x28, 0x05, 0x52, 0x02,
	0x72, 0x78, 0x12, 0x0e, 0x0a, 0x02, 0x72, 0x79, 0x18, 0x05, 0x20, 0x01, 0x28, 0x05, 0x52, 0x02,
	0x72, 0x79, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x18, 0x07, 0x20, 0x01, 0x28,
	0x02, 0x52, 0x05, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x22, 0x62, 0x0a, 0x0e, 0x46, 0x61, 0x63, 0x65,
	0x44, 0x65, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x1c, 0x0a, 0x09, 0x69, 0x6d,
	0x61, 0x67, 0x65, 0x44, 0x61, 0x74, 0x61, 0x18, 0x01, 0x20, 0x01, 0x28, 0x0c, 0x52, 0x09, 0x69,
	0x6d, 0x61, 0x67, 0x65, 0x44, 0x61, 0x74, 0x61, 0x12, 0x12, 0x0a, 0x04, 0x74, 0x79, 0x70, 0x65,
	0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x74, 0x79, 0x70, 0x65, 0x12, 0x1e, 0x0a, 0x0a,
	0x63, 0x6f, 0x6e, 0x66, 0x54, 0x68, 0x72, 0x65, 0x73, 0x68, 0x18, 0x03, 0x20, 0x01, 0x28, 0x02,
	0x52, 0x0a, 0x63, 0x6f, 0x6e, 0x66, 0x54, 0x68, 0x72, 0x65, 0x73, 0x68, 0x22, 0x47, 0x0a, 0x0f,
	0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12,
	0x34, 0x0a, 0x07, 0x64, 0x65, 0x74, 0x4f, 0x62, 0x6a, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b,
	0x32, 0x1a, 0x2e, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e,
	0x2e, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74, 0x65, 0x64, 0x4f, 0x62, 0x6a, 0x52, 0x07, 0x64, 0x65,
	0x74, 0x4f, 0x62, 0x6a, 0x73, 0x32, 0x5c, 0x0a, 0x0e, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74,
	0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x4a, 0x0a, 0x07, 0x70, 0x72, 0x65, 0x64, 0x69,
	0x63, 0x74, 0x12, 0x1d, 0x2e, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69,
	0x6f, 0x6e, 0x2e, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73,
	0x74, 0x1a, 0x1e, 0x2e, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f,
	0x6e, 0x2e, 0x46, 0x61, 0x63, 0x65, 0x44, 0x65, 0x74, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x22, 0x00, 0x42, 0x4b, 0x0a, 0x25, 0x63, 0x6f, 0x6d, 0x2e, 0x62, 0x79, 0x74, 0x65, 0x64,
	0x61, 0x6e, 0x63, 0x65, 0x2e, 0x76, 0x69, 0x64, 0x65, 0x6f, 0x61, 0x72, 0x63, 0x68, 0x2e, 0x66,
	0x61, 0x63, 0x65, 0x64, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0x42, 0x09, 0x46, 0x61,
	0x63, 0x65, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x50, 0x01, 0x5a, 0x0f, 0x2e, 0x3b, 0x66, 0x61, 0x63,
	0x65, 0x64, 0x65, 0x74, 0x65, 0x63, 0x74, 0x69, 0x6f, 0x6e, 0xa2, 0x02, 0x03, 0x52, 0x54, 0x47,
	0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_facedetection_face_detection_proto_rawDescOnce sync.Once
	file_facedetection_face_detection_proto_rawDescData = file_facedetection_face_detection_proto_rawDesc
)

func file_facedetection_face_detection_proto_rawDescGZIP() []byte {
	file_facedetection_face_detection_proto_rawDescOnce.Do(func() {
		file_facedetection_face_detection_proto_rawDescData = protoimpl.X.CompressGZIP(file_facedetection_face_detection_proto_rawDescData)
	})
	return file_facedetection_face_detection_proto_rawDescData
}

var file_facedetection_face_detection_proto_msgTypes = make([]protoimpl.MessageInfo, 3)
var file_facedetection_face_detection_proto_goTypes = []interface{}{
	(*DetectedObj)(nil),     // 0: FaceDetection.DetectedObj
	(*FaceDetRequest)(nil),  // 1: FaceDetection.FaceDetRequest
	(*FaceDetResponse)(nil), // 2: FaceDetection.FaceDetResponse
}
var file_facedetection_face_detection_proto_depIdxs = []int32{
	0, // 0: FaceDetection.FaceDetResponse.detObjs:type_name -> FaceDetection.DetectedObj
	1, // 1: FaceDetection.FaceDetService.predict:input_type -> FaceDetection.FaceDetRequest
	2, // 2: FaceDetection.FaceDetService.predict:output_type -> FaceDetection.FaceDetResponse
	2, // [2:3] is the sub-list for method output_type
	1, // [1:2] is the sub-list for method input_type
	1, // [1:1] is the sub-list for extension type_name
	1, // [1:1] is the sub-list for extension extendee
	0, // [0:1] is the sub-list for field type_name
}

func init() { file_facedetection_face_detection_proto_init() }
func file_facedetection_face_detection_proto_init() {
	if File_facedetection_face_detection_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_facedetection_face_detection_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DetectedObj); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_facedetection_face_detection_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*FaceDetRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_facedetection_face_detection_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*FaceDetResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_facedetection_face_detection_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   3,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_facedetection_face_detection_proto_goTypes,
		DependencyIndexes: file_facedetection_face_detection_proto_depIdxs,
		MessageInfos:      file_facedetection_face_detection_proto_msgTypes,
	}.Build()
	File_facedetection_face_detection_proto = out.File
	file_facedetection_face_detection_proto_rawDesc = nil
	file_facedetection_face_detection_proto_goTypes = nil
	file_facedetection_face_detection_proto_depIdxs = nil
}