import os
import onnx
import numpy as np
import sys


def Usage():
    print("Usage:")
    print("\tpython ", sys.argv[0], " <input pb folder> <output folder>")





def getFileNames(dir: str):
    files = os.listdir(dir)
    #ret = [os.path.join(dir,x) for x in files if x.endswith('tensorproto')]
    return files


def __onnx_type_to_numpy_type(onnx_type):
    mapping = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT16: np.uint16,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.COMPLEX64: np.complex64,
        onnx.TensorProto.COMPLEX128: np.complex128,
        onnx.TensorProto.UINT32: np.uint32,
        onnx.TensorProto.UINT64: np.uint64,
    }
    return mapping.get(onnx_type, None)


def convertSingle(inputFile, outputFile, T=False):
    print(inputFile, outputFile)

    tensor_data = ""
    tensor_proto = onnx.TensorProto()
    with open(inputFile, 'rb') as f:
        tensor_data = f.read()
    with open(inputFile, 'rb') as f:
        tensor_proto.ParseFromString(f.read())

    data_type_str = onnx.TensorProto.DataType.Name(tensor_proto.data_type)

    if len(tensor_proto.dims) != 4:
        with open(outputFile, 'wb') as f:
            f.write(tensor_proto.raw_data)
        print(
            f'{inputFile} data_type is {data_type_str}, dims is{tensor_proto.dims}'
        )
    else:
        if T:
            data_type = tensor_proto.data_type
            raw_data = tensor_proto.raw_data
            shape = tensor_proto.dims
            np_type = __onnx_type_to_numpy_type(tensor_proto.data_type)
            # Convert raw_data to a NumPy array based on the data_type
            if np_type:
                numpy_array = np.frombuffer(raw_data,
                                            dtype=np_type).reshape(shape)
                numpy_array = np.transpose(numpy_array, axes=(0, 2, 3, 1))
                numpy_array.tofile(outputFile)
                print(
                    f'{inputFile} data_type is {data_type_str}, new dims is{numpy_array.shape}'
                )
            else:
                print(f'not support data_type : {data_type_str}')
                exit(-1)
        else:
            with open(outputFile, 'wb') as f:
                f.write(tensor_proto.raw_data)
            print(
                f'{inputFile} data_type is {data_type_str}, dims is{tensor_proto.dims}'
            )
    pass


def createIfNotExist(outputDir):
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)


def convert_pb_to_bin(src,dst):
    createIfNotExist(dst)
    allFileNames = getFileNames(src)

    for pbFileName in allFileNames:
        fn, ext = os.path.splitext(pbFileName)
        convertSingle(os.path.join(src, pbFileName),
                      os.path.join(dst, fn + ".bin"))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        Usage()
        print("use default path:dump,dump_bin")
        sys.argv = [sys.argv[0],"dump","dump_bin"]
    inputDir = sys.argv[1]
    outputDir = sys.argv[2]
    convert_pb_to_bin(inputDir,outputDir)
