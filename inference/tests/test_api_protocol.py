import json
import unittest
from acc_infer_clear.api.protocol import parse_event, audio_event


class ProtocolTests(unittest.TestCase):
    def test_valid_requests(self):
        for event in ({"op":"open","id":"a","seed":2,"emotion":[0]*8},
                      {"op":"text","id":"a","text":"测试"}, {"op":"end","id":"a"},
                      {"op":"cancel","id":"a"}, {"op":"release","id":"a"},
                      {"op":"run"}, {"op":"tick"}, {"op":"drain"}):
            self.assertEqual(parse_event(json.dumps(event)), event)

    def test_invalid_requests(self):
        for event in ([], {}, {"op":"unknown"}, {"op":"run","id":"a"},
                      {"op":"open","id":None}, {"op":"open","id":""},
                      {"op":"open","id":"a","seed":True},
                      {"op":"open","id":"a","seed":2**63},
                      {"op":"text","id":"a","text":42},
                      {"op":"open","id":"a","emotion":[float("nan")]*8}):
            with self.assertRaises(ValueError):
                parse_event(json.dumps(event))

    def test_audio_layout(self):
        import base64
        import numpy as np
        pcm=np.array([-32768,0,32767],dtype=np.int16)
        result=audio_event(dict(request_id="a",complete=True,chunk=dict(
            pcm=pcm,index=0,sample_start=0,sample_end=3)))
        self.assertEqual(base64.b64decode(result["pcm_s16le"]),pcm.astype("<i2").tobytes())
        self.assertEqual(result["sample_rate"],22050)


if __name__ == "__main__":
    unittest.main()
