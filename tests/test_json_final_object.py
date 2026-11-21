import json
import pytest
from streaming.flat_speaking import _json_object,SegmentRejected


def test_complete_final_correction_preserves_exact_authored_fields():
    final={'text':'Authored prefix.','draft':'Authored remaining speech.','framework':{'ready':True}}
    raw='{"text":"invalid",\",\"framework":{}}\nHere is the clean final object:\n\n'+json.dumps(final)
    assert _json_object(raw)==final


@pytest.mark.parametrize('raw',[
    '{"text":"prefix","draft":"body"},"framework":{"ready":true}}',
    'broken\n{"text":"valid object"}\nextra commentary',
    'broken\n{"text":"unterminated',
    '[\n{"text":"inner array element"}\n]',
])
def test_incomplete_or_nonfinal_fragments_remain_rejected(raw):
    with pytest.raises(SegmentRejected):_json_object(raw)
