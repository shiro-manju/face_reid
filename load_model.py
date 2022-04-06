#
# 下記のコードを参考にさせて頂きました。
# https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/python_demos/asl_recognition_demo/asl_recognition_demo/common.py
#

class Model:
    def __init__(self, model_path, device, ie_core, num_requests, output_shape=None):
        if model_path.endswith((".xml", ".bin")):
            model_path = model_path[:-4]
        print('model_path: {}'.format(model_path))
        self.net = ie_core.read_network(model_path + ".xml", model_path + ".bin")
        assert len(self.net.input_info) == 1, "One input is expected"

        '''
        supported_layers = ie_core.query_network(self.net, device)
        not_supported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) > 0:
            raise RuntimeError("Following layers are not supported by the {} plugin:\n {}"
                               .format(device, ', '.join(not_supported_layers)))
        '''

        self.exec_net = ie_core.load_network(network=self.net,
                                             device_name=device,
                                             num_requests=num_requests)

        self.input_name = next(iter(self.net.input_info))
        if len(self.net.outputs) > 1:
            if output_shape is not None:
                candidates = []
                for candidate_name in self.net.outputs:
                    candidate_shape = self.exec_net.requests[0].output_blobs[candidate_name].buffer.shape
                    if len(candidate_shape) != len(output_shape):
                        continue

                    matches = [src == trg or trg < 0
                               for src, trg in zip(candidate_shape, output_shape)]
                    if all(matches):
                        candidates.append(candidate_name)

                if len(candidates) != 1:
                    raise Exception("One output is expected")

                self.output_name = candidates[0]
            else:
                raise Exception("One output is expected")
        else:
            self.output_name = next(iter(self.net.outputs))

        self.input_size = self.net.input_info[self.input_name].input_data.shape
        self.output_size = self.exec_net.requests[0].output_blobs[self.output_name].buffer.shape
        self.num_requests = num_requests

    def infer(self, data):
        input_data = {self.input_name: data}
        
        infer_result = self.exec_net.infer(input_data)
        return infer_result[self.output_name]