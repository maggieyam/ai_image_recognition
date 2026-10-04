"""
Eval for recognizing how people feel, end to end (recognize_feelings.how_people_feel):
picture -> Florence-2 description -> does it mention a person? -> Florence-2
finds the main characters -> the emotion model reads each face -> Qwen puts
them, as JSON, into words.

Each case is a real picture with the expected outcome: an emotion the answer
must name, or NOTHING (no feelings said: animals, objects, empty scenes,
statues, background crowds, faces that can't be seen). Failures are blamed on
the stage that caused them.

The pictures aren't in the repo. OASIS goes in data/oasis/ (see CLAUDE.md:
datasets are never committed); the app's sample photos in images/. Cases
whose picture is missing are skipped. Results are saved to data/evals/.

    python evals/recognize_feelings_eval.py
"""
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

import recognize_feelings  # noqa: E402
from llm import MODEL_FILE, Qwen  # noqa: E402
from emotions import EMOTION_MODEL, EmotionRecognizer  # noqa: E402
from recognizer import FLORENCE_ID, ImageRecognizer  # noqa: E402

NOTHING = None
OASIS = ROOT / "data" / "oasis" / "Images"
SAMPLES = ROOT / "images"


def oasis(names, expected):
    return [(OASIS / f"{n}.jpg", expected) for n in names]


def numbered(theme, count, expected):
    return oasis([f"{theme} {i}" for i in range(1, count + 1)], expected)


# (group, [(picture, expected)]). OASIS face themes give the expected emotion;
# the NOTHING pictures were checked by eye.
CASES = {
    "people": numbered("Angry face", 5, "Anger")
    + numbered("Sad face", 9, "Sadness")
    + numbered("Depressed face", 2, "Sadness")
    + numbered("Miserable face", 2, "Sadness")
    + numbered("Happy face", 2, "Happiness")
    + numbered("Smiling face", 1, "Happiness")
    + numbered("Excited face", 7, "Happiness")
    + numbered("Scared face", 5, "Fear")
    + numbered("Neutral face", 5, "Neutral")
    + numbered("Surprise", 2, "Surprise"),
    "look-alikes": oasis(["Statue 1", "Statue 2", "Gargoyle 1", "Gargoyle 2", "Rubber duck 1"], NOTHING),
    "animals": oasis(
        ["Dog 1", "Cat 1", "Bear 1", "Lion 1", "Monkey 1", "Shark 1", "Snake 1", "Bird 1", "Horse 1", "Zebra 1"],
        NOTHING,
    ),
    "objects": oasis(
        ["Acorns 1", "Alcohol 1", "Cups 1", "Dessert 1", "Flowers 1", "Keyboard 1", "Rocks 1", "Toilet 1",
         "Yarn 1", "Fire hydrant 1", "Dummy 1"],
        NOTHING,
    ),
    "empty scenes": oasis(
        ["Lake 1", "Galaxy 1", "Lightning 1", "Waterfall 1", "Sunset 1", "Snow 1", "Thunderstorm 1",
         "Volcano 1", "Rainbow 1", "Desert 1"],
        NOTHING,
    ),
    # The app's sample photos: strawberries, bears, a guitarist walking away
    # (no face to read), a desk with figurines, a station crowd (background).
    "sample photos": [(SAMPLES / f"photo_{i}.jpg", NOTHING) for i in range(1, 6)],
}
# Words that count as naming each of the face model's emotions.
WORDS = {
    "Anger": ["anger", "angry", "furious"],
    "Sadness": ["sad", "unhappy", "depressed", "miserable", "sorrow"],
    "Happiness": ["happy", "happiness", "joy", "smil", "excited", "cheerful"],
    "Fear": ["fear", "afraid", "scared", "frightened", "terrified"],
    "Neutral": ["neutral", "calm", "expressionless"],
    "Surprise": ["surprise", "astonish", "shock"],
}


def run_case(recognizer, emotions, qwen, picture, expected):
    image = Image.open(picture).convert("RGB")
    _, description = recognizer.describe(image)
    result = recognize_feelings.how_people_feel(qwen, recognizer, emotions, image, description)
    answer, people = result["answer"], result["people"]
    person = recognize_feelings.mentions_a_person(description)
    found = [p["emotion"] for p in people]
    if expected is NOTHING:
        ok = answer is None
        stage = None if ok else "spoke about feelings"
    else:
        ok = any(word in (answer or "").lower() for word in WORDS[expected])
        if ok:
            stage = None
        elif not person:
            stage = "description mentions no person"
        elif not people:
            stage = "no main-character face found"
        elif expected not in found:
            stage = "face model read another emotion"
        else:
            stage = "Qwen's wording"
    return {
        "picture": picture.name, "expected": expected, "description": description,
        "mentions_person": person, "faces": found, "humans": [p["human"] for p in people], "answer": answer, "ok": ok, "failed_at": stage,
    }


def main():
    recognizer, emotions, qwen = ImageRecognizer(), EmotionRecognizer(), Qwen()
    results = defaultdict(list)
    for group, cases in CASES.items():
        print(f"\n{group}")
        for picture, expected in cases:
            if not picture.exists():
                print(f"  skip {picture.name} (not found)")
                continue
            r = run_case(recognizer, emotions, qwen, picture, expected)
            results[group].append(r)
            print(f"  {'ok  ' if r['ok'] else 'FAIL'} {r['picture']:22} expected {expected or 'nothing':10}"
                  f" person={r['mentions_person']!s:5} faces={r['faces']}  {r['failed_at'] or ''}\n"
                  f"       {r['answer']}", flush=True)

    print("\nSummary")
    for group, rs in results.items():
        print(f"  {group:14} {sum(r['ok'] for r in rs)}/{len(rs)}")
    failures = Counter(r["failed_at"] for rs in results.values() for r in rs if not r["ok"])
    for stage, n in failures.most_common():
        print(f"  failed at {stage}: {n}")

    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    out = ROOT / "data" / "evals" / f"recognize-feelings-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "commit": git("rev-parse", "--short", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain")),
        "models": {"describe": FLORENCE_ID, "faces": EMOTION_MODEL, "qwen": MODEL_FILE},
        "results": results,
    }, indent=1))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
