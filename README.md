# Image Recognizer

Find out what's in a picture. Upload an image or paste a screenshot, and get a short plain-English description.

## How to use

1. Upload an image, or paste one with Ctrl/Cmd+V.
2. Click **Describe image**. What the app says appears a part at a time: what it saw, how the people in the picture look (if there are any), how the picture makes it feel, with its top three moods, and an offer to write you a story. On a laptop the first part takes about 15 seconds.
3. Optionally pick a mood and a genre and click **Brainstorm story ideas** for 1–3 short story pitches. Expect about half a minute, plus 15–30 seconds per idea.
4. Optionally press **Speak out loud** to hear it as well. The button is only there when you run the app yourself on a computer with a speaker, and the voice comes out of that computer.

## Good to know

- **It can be wrong.** Descriptions are one-sentence captions, and can miss small objects or who is doing what.
- **How people feel is read from their faces**, and only the main characters'. It's a guess about how someone looks, not a fact about how they feel.
- **Moods are the app's own reaction**, picked from a fixed list. It can feel differently about a picture than you do.
- **Story ideas are made up**, starting from the picture and the mood you picked.
- **Images:** common formats such as JPG and PNG, up to 16 MB. They aren't saved.
- **Free demo limits:** 10 requests per minute and 100 per day per visitor. The first request after a quiet period can be slow.

## Run it yourself

Everything runs on your own computer, so there are no limits and your images never leave it.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000. Installing compiles llama.cpp, which takes a few minutes, and the first run downloads about 2.3 GB of model weights. A GPU is used if you have one.

> **Intel Mac:** `requirements.txt` pins `transformers<5` and `numpy<2`, because PyTorch's last build for Intel Macs is 2.2.2.

To serve it to other people, use gunicorn with one worker, since each worker loads its own copy of the models:

```bash
SECRET_KEY=change-me gunicorn -w 1 --threads 4 -b 0.0.0.0:8000 app:app
```

| Setting | What it does |
|---|---|
| `SECRET_KEY` | Signs the story ideas the server hands out. Without it, ideas planned before a restart can't be written. |
| `SPEAK_ALOUD=1` | Lets a served app speak through its machine's speaker, for a robot, say. It's off under gunicorn otherwise, because any visitor could make that machine talk. |
| `VOICE` | The voice it speaks in: `piper`, the only one so far. |
| `PORT` | The port `python app.py` listens on (8000). |

On Linux, speaking out loud needs PortAudio: `sudo apt install libportaudio2`.

- **Tests** (no models needed): `pip install pytest && pytest`
- **Command line:** `python -m brain images/*.jpg` prints each image's caption, moods and description.

## How it works

| Job | Model | Code |
|---|---|---|
| Describing the picture | [Florence-2](https://huggingface.co/florence-community/Florence-2-base) | `brain/perception/sight.py` |
| Feeling a mood | [CLIP](https://huggingface.co/openai/clip-vit-base-patch32), zero-shot | `brain/affect/mood.py` |
| Finding the main characters' faces | [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `brain/perception/faces.py` |
| Reading each face | [EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib) | `brain/networks/face_emotion.py` |
| Putting it into words, and story ideas | [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF), 4-bit, on [llama.cpp](https://github.com/abetlen/llama-cpp-python) | `brain/language/speech.py`, `brain/imagination/story.py` |
| Speaking out loud | [Piper](https://github.com/OHF-Voice/piper1-gpl), in a voice trained on public-domain [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) | `brain/networks/piper.py`, `effectors/mouth.py` |

Qwen never sees the picture: it only words what the other models found. Story ideas are brainstormed from Florence-2's description, checked against the picture with CLIP, then written one at a time.

## How the code is organised

The code is laid out like the parts of a mind, so a new ability can be added without changing the others.

```
senses/          eye.py                    takes in the picture; no model, no understanding
effectors/       mouth.py                  says things through the speaker; no model, no language
brain/
  workspace.py                             what the brain has in mind about one picture
  faculty.py                               what every faculty looks like, and running them in turn
  brain.py                                 loads the networks, lists the faculties
  networks/                                the models, each loaded once; voices.py lists the voices
  perception/    sight.py  faces.py        what's in the picture, who, and how they look
  affect/        mood.py                   how the picture makes the app feel
  language/      speech.py  sentences.py   putting what it has to say into words
  imagination/   story.py                  story ideas
app.py                                     the web server: requests, limits, signed ideas
web/                                       the page
```

Each **faculty** is one specialist (`Sight`, `ReadFaces`, `Mood`, `Speak`). It declares what it `needs` and what it `gives`, and faculties never call each other: they read from and post to a shared **workspace**. A faculty is skipped when something it needs isn't there.

To add an ability, write a faculty and add it to `build_faculties` in `brain/brain.py`:

```python
class Mood(Faculty):
    needs = ("image",)
    gives = ("moods",)

    def run(self, workspace):
        workspace.post(moods=self.moods(workspace["image"]))
```

To add a voice, write a class with a `sounds(text)` method in `brain/networks/` and list it in `brain/networks/voices.py`. `VOICE=<its name>` then switches to it.

The layout borrows from [global workspace theory](https://en.wikipedia.org/wiki/Global_workspace_theory). It's an analogy to organise code by, not a model of a brain: faculties run in a fixed order, and nothing is remembered between pictures.

## Example
<img width="614" height="896" alt="image" src="https://github.com/user-attachments/assets/e27ef025-d9ee-419f-90c3-6d47641ceed6" />
