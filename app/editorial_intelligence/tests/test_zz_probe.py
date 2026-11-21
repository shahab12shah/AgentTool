import pytest
from app.editorial_intelligence.context import build_editorial_context
from app.editorial_intelligence.tests.scenarios import V, craft
from app.editorial_intelligence.tests.conftest import *  # noqa

NARR = {
 "edu": "Why does the price of silver keep rising? Silver is used in solar panels and electronics. Manufacturers consume more of the metal every year. According to the official report, demand grew 20% in 2023. Compared to gold, silver is far more volatile. As a result, prices jumped again. In conclusion, silver demand is driven by industry. Remember to subscribe for more.",
 "inv": "The investigation begins. Police interview witnesses. A previously overlooked piece of evidence appears.",
}

@pytest.mark.parametrize("name", ["edu", "inv"])
def test_probe(ws, tmp_path, name):
    craft(ws, tmp_path, NARR[name], {}, min_scene=1.0, threshold=0.15)
    ctx = build_editorial_context(ws.project)
    print()
    for s in ctx.scenes:
        sc = s.sc
        print(s.id, sc.scene.label, round(sc.scene.start,1), round(sc.scene.end,1), repr(sc.scene.narration), "topic=", sc.scene.topic, "imp=", round(sc.scene.importance,2))
        print("   ents", [(e.canonical, e.type.value) for e in sc.scene.entities], "new", [e.canonical for e in sc.new_entities], "claims", [(c.type.value, c.text[:30]) for c in sc.scene.claims], "nums", [n.text for n in sc.scene.numbers])
        for sid_, a in sc.sentence_analysis.items():
            print("   sent", sid_, a.cues, a.text[:50], a.terms[:5])
        print("   starts_section", sc.starts_section, "intent", sc.intent.type.value if sc.intent else None, "inh", sc.intent.inherited_from if sc.intent else None)
