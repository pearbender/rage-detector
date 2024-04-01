from faster_whisper import WhisperModel
import torch

model = WhisperModel(
    'large-v2', device="cuda" if torch.cuda.is_available() else "cpu")

segments, info = model.transcribe('test.wav',
                                  language='ja',
                                  initial_prompt="PearBender welcome, fuck english... ええと こんばんは、Alex welcome, いやねぇ 今日はねぇ あ ところでさ ごめん あの 今日ねぇ 今日ねぇ みんな 今日ねぇ, cha- cha- what is it? cha- chazay? あとなんか変な味がする… 口の中, Pero welcome, Mathew welcome, Ender welcome. I'm playing wa- i don't get it wa- warhammer",
                                  beam_size=3,
                                  best_of=3,
                                  temperature=(-2.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                                  suppress_tokens=[],
                                  vad_filter=True,
                                  vad_parameters=dict(
                                      min_silence_duration_ms=800)
                                  )

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
