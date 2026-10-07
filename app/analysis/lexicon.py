"""Hand-written lexicons for the rule-based analyzer.

These are deliberately small and conservative. They are the knowledge base of the *heuristic*
analyzer, not a substitute for a language model: unknown entities are only found through
capitalisation/script casing, and many real-world names will be missed. A model-backed
``SemanticAnalyzer`` can replace all of this behind the same interface.
"""

from __future__ import annotations

from app.analysis.models import EntityType as E
from app.core.textutil import stem


def _phrases(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


_LEX_SOURCES: dict[E, str] = {
    E.GOVERNMENT_AGENCY: (
        "IRS, SEC, FDA, FBI, CIA, NASA, EPA, FTC, FCC, DOJ, CDC, NIH, TSA, DHS, FEMA, OSHA, ATF, DEA, CFPB, FDIC, "
        "Federal Reserve, the Fed, Treasury, Department of Justice, Department of Energy, Department of Labor, "
        "Social Security Administration, Internal Revenue Service, Congress, Senate, White House, Pentagon, Supreme Court, "
        "European Commission, Bank of England, European Central Bank, ECB, Medicare, Medicaid"
    ),
    E.COUNTRY: (
        "United States, USA, America, United Kingdom, UK, Britain, England, Canada, Mexico, Brazil, Argentina, Chile, Peru, "
        "China, Japan, India, Russia, Ukraine, Germany, France, Italy, Spain, Portugal, Netherlands, Belgium, Sweden, Norway, "
        "Denmark, Finland, Poland, Turkey, Greece, Israel, Iran, Iraq, Saudi Arabia, Egypt, Nigeria, Kenya, South Africa, "
        "Australia, New Zealand, South Korea, North Korea, Taiwan, Vietnam, Thailand, Indonesia, Pakistan, Switzerland, Ireland"
    ),
    E.CITY: (
        "New York, Los Angeles, Chicago, Houston, Washington, San Francisco, Boston, Seattle, Miami, Dallas, Las Vegas, "
        "London, Paris, Berlin, Rome, Madrid, Moscow, Beijing, Shanghai, Hong Kong, Tokyo, Seoul, Singapore, Dubai, Mumbai, "
        "Delhi, Sydney, Toronto, Mexico City, Cairo, Istanbul, Wall Street, Silicon Valley"
    ),
    E.COMPANY: (
        "Apple, Microsoft, Google, Alphabet, Amazon, Meta, Facebook, Tesla, Nvidia, Intel, AMD, Samsung, Sony, Toyota, Ford, "
        "General Motors, Boeing, Airbus, Walmart, Costco, Target, Netflix, Disney, Intuit, TurboTax, Coca-Cola, Pepsi, "
        "JPMorgan, Goldman Sachs, Morgan Stanley, Bank of America, Wells Fargo, Citigroup, BlackRock, Vanguard, Fidelity, "
        "Berkshire Hathaway, Visa, Mastercard, PayPal, Uber, Airbnb, Spotify, OpenAI, Anthropic, Oracle, IBM, Cisco, "
        "Exxon, Chevron, Shell, BP, First Solar, SunPower, Tata, Alibaba, Tencent, Huawei, TSMC"
    ),
    E.FINANCIAL_INSTRUMENT: (
        "silver, gold, platinum, palladium, copper, oil, crude oil, natural gas, bitcoin, ethereum, crypto, cryptocurrency, "
        "stocks, stock, shares, bonds, bond, treasuries, ETF, ETFs, mutual funds, index funds, futures, options, derivatives, "
        "dollar, the dollar, euro, yen, 401k, IRA, Roth IRA, S&P 500, Dow Jones, Nasdaq, commodities, real estate, mortgage"
    ),
    E.TECHNOLOGY: (
        "solar panels, solar panel, solar, solar cells, solar power, wind turbines, batteries, battery, lithium, semiconductors, "
        "semiconductor, microchips, chips, artificial intelligence, AI, machine learning, blockchain, robotics, robots, drones, "
        "electric vehicles, electric vehicle, EVs, EV, self-driving cars, 5G, quantum computing, cloud computing, "
        "nuclear power, hydrogen, fusion, 3D printing, software, algorithms, data centers, electronics"
    ),
    E.OBJECT: (
        "car, cars, truck, trucks, house, houses, home, homes, building, buildings, bridge, factory, factories, warehouse, "
        "computer, computers, laptop, phone, phones, smartphone, smartphones, camera, television, tv, coin, coins, bars, "
        "jewelry, ring, rings, watch, document, documents, letter, notice, form, forms, check, checks, contract, passport, "
        "package, packages, ship, ships, plane, planes, train, trains, bottle, tools, machine, machines, pipe, pipes, "
        "wire, wires, mirror, mirrors, cell phone, circuit boards, circuit board, mine, mines, vault, safe, wallet"
    ),
}

HONORIFICS = {"mr", "mrs", "ms", "dr", "prof", "president", "senator", "governor", "ceo", "chairman", "secretary", "judge", "minister"}
FIRST_NAMES = set(_phrases(
    "james, john, robert, michael, william, david, richard, joseph, thomas, charles, mary, patricia, jennifer, linda, elizabeth, "
    "barbara, susan, jessica, sarah, karen, nancy, lisa, daniel, matthew, anthony, mark, donald, steven, paul, andrew, joshua, "
    "kenneth, kevin, brian, george, edward, ronald, timothy, jason, jeffrey, ryan, jacob, gary, nicholas, eric, jonathan, "
    "stephen, larry, justin, scott, brandon, benjamin, samuel, gregory, alexander, frank, raymond, jack, dennis, jerry, tyler, "
    "aaron, jose, adam, henry, nathan, douglas, zachary, peter, kyle, walter, ethan, jeremy, harold, keith, christian, roger, "
    "noah, gerald, carl, terry, sean, austin, arthur, lawrence, jesse, dylan, bryan, joe, jordan, billy, bruce, albert, willie, "
    "gabriel, logan, alan, juan, wayne, roy, ralph, randy, eugene, vincent, russell, elijah, louis, bobby, philip, johnny, "
    "elon, warren, jeff, bill, steve, tim, sam, janet, jerome, powell, yellen, anna, emily, olivia, sophia, emma, ava, isabella"
))
ORG_SUFFIXES = {"inc", "corp", "corporation", "company", "co", "ltd", "llc", "group", "association", "institute", "foundation",
                "university", "bank", "agency", "department", "committee", "council", "commission", "authority", "union", "fund"}

def _build() -> dict[tuple[str, ...], E]:
    from app.core.textutil import tokenize

    out: dict[tuple[str, ...], E] = {}
    for etype, src in _LEX_SOURCES.items():
        for phrase in _phrases(src):
            key = tuple(t.norm for t in tokenize(phrase))
            if key and key not in out:
                out[key] = etype
    return out


GAZETTEER: dict[tuple[str, ...], E] = _build()
MAX_PHRASE = max(len(k) for k in GAZETTEER)
# Single-token entries that must be capitalised to count (genuinely ambiguous words).
STRICT_CASE = {"apple", "amazon", "target", "meta", "oracle", "shell", "visa", "fed", "congress", "bp"}

# ---------------------------------------------------------------- cue words (stems)
def _stems(text: str) -> frozenset[str]:
    return frozenset(stem(w) for w in text.split())


CUES: dict[str, frozenset[str]] = {
    "process": _stems("manufacturer manufacturers manufacture manufacturing producer producers produce production build building make making convert converting install installing assemble "
                      "mine mining refine refining process processing consume consuming consumption use using used apply generate generating "
                      "deliver delivering grow growing extract extracting recycle recycling transform operate work works pipeline supply chain"),
    "data": _stems("price cost costs percent percentage rate growth increase increased decrease decreased rose rise fell fall climbed dropped "
                   "surged jumped plunged billion million trillion thousand statistic statistics data average ratio total chart revenue profit "
                   "inflation yield worth valued expensive cheap"),
    "evidence": _stems("document notice letter report study filing filed according official statement memo email evidence record contract audit "
                       "form announcement press release published survey poll research paper court ruling ruled law act rule regulation "
                       "legislation bill statute policy"),
    "person": _stems("ceo president senator governor founder owner investor investors customer customers worker workers people taxpayer "
                     "taxpayers retiree retirees family families consumer consumers employee employees citizen citizens man woman child "
                     "children voters buyers sellers"),
    "location": _stems("city country region border state states capital abroad overseas continent island coast area nation"),
    "comparison": _stems("than versus compared compare comparison while whereas unlike both either rather difference different better worse "
                         "outperform instead similar contrast"),
    "event": _stems("announced announce crashed crash collapsed launched launch signed happened occurred started ended war election crisis "
                    "recession meeting summit attack bankrupt merger acquisition passed approved banned arrested investigation scandal "
                    "protest strike"),
    "abstract": _stems("idea concept future risk important importance impact trend change shift uncertain uncertainty freedom fear success "
                       "strategy pressure opportunity challenge problem solution question reason matter mean means value"),
    "emotional": _stems("shocking shocked crisis devastating huge massive incredible dangerous terrifying alarming critical urgent worst best "
                        "amazing unbelievable disaster panic fear outrage furious"),
    "conclusion": _stems("conclusion conclude overall ultimately finally summary takeaway lesson remember bottom line"),
    "prediction": _stems("will could may might expected forecast predict projected likely probably soon eventually upcoming future going"),
    "opinion": _stems("think believe feel opinion seems should better worse best worst terrible great amazing love hate"),
    "law": _stems("law act statute legislation bill ruling court regulation legal illegal lawsuit constitution amendment code"),
    "rule": _stems("rule requirement required requires must eligible eligibility deadline policy qualify allowed prohibited mandatory penalty"),
    "quote": _stems("said says say told stated according quoted announced wrote tweeted claimed argued"),
    "price": _stems("price priced cost costs trading traded worth reached hit surged valued quote"),
    "deadline": _stems("deadline due before until by expires expire expiry"),
    "age": _stems("age aged old retirement retire senior"),
    "domain_market": _stems("price market demand supply stock shares investor trading commodity silver gold oil inflation economy economic "
                            "rates fund etf bond yield dollar currency inflation recession"),
    "domain_legal": _stems("law court legal tax taxes irs regulation legislation penalty"),
}

TRANSITIONS_STRONG = ("moving on", "next up", "next,", "meanwhile", "let's talk about", "lets talk about", "let's move", "now let's",
                      "now lets", "in conclusion", "to sum up", "finally", "lastly", "here's the thing", "so what", "on the other hand",
                      "let's look at", "lets look at", "turning to", "speaking of")
TRANSITIONS_WEAK = ("now", "next", "however", "anyway", "first", "second", "third", "another", "okay", "so", "but what", "instead",
                    "today", "welcome", "let's", "lets")
CONTINUATIONS = ("it", "this", "that", "these", "those", "they", "he", "she", "its", "their", "and", "also", "because", "which",
                 "as a result", "therefore", "that's why", "thats why", "plus", "in fact", "for example", "for instance", "such as",
                 "which means", "meaning", "then", "additionally", "furthermore", "and that", "and this", "not only")
QUESTION_STARTS = ("what", "why", "how", "who", "when", "where", "which", "is", "are", "do", "does", "did", "can", "could", "should",
                   "would", "will", "have", "has", "was", "were")
PROCESS_VERBS = _stems("manufacture produce build make convert install assemble mine refine process consume use apply generate deliver grow "
                       "extract recycle transform operate power store transport sell buy invest trade file pay send receive")
