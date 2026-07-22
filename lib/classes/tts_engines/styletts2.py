from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets

# StyleTTS2 ukrainian (patriotyk). Unlike the coqui engines this one is phoneme
# driven: ukrainian stress is predicted first, then converted to IPA, and only
# then tokenized. Skipping the stress step is what makes every other ukrainian
# engine here mispronounce homographs (за́мок / замо́к).

_stressify = None


class StyleTTS2(TTSUtils, TTSRegistry, name='styletts2'):

    def __init__(self, session:DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speakers_path = default_engine_settings[TTS_ENGINES['STYLETTS2']]['speakers_path']
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.tts_zs_key = default_vc_model.rsplit('/',1)[-1]
            self.pth_voice_file = None
            self.resampler_cache = {}
            self.resampled_wav_cache = {}
            self.audio_segments = []
            self.style_cache = {}
            self.models = load_engine_presets(self.session['tts_engine'])
            self.params = {"semitones": {}, "samplerate": None}
            # effective language for TTS (target when translating, else source)
            self.language = self.session.get('language')
            self.language_iso1 = self.session.get('language_iso1')
            if self.session.get('translate_enabled'):
                if self.session.get('translate'):
                    self.language = self.session['translate']
                if self.session.get('translate_iso1'):
                    self.language_iso1 = self.session['translate_iso1']
            tts_engine = self.session.get('tts_engine')
            if tts_engine not in default_engine_settings:
                error = f'Invalid tts_engine {tts_engine}.'
                raise ValueError(error)
            engine_langs = default_engine_settings[tts_engine].get('languages', {})
            if self.language not in engine_langs:
                error = f'Language {self.language} not supported by engine {tts_engine}.'
                raise ValueError(error)
            fine_tuned = self.session.get('fine_tuned')
            if fine_tuned not in self.models:
                error = f'Invalid fine_tuned model {fine_tuned}. Available models: {list(self.models.keys())}'
                raise ValueError(error)
            model_cfg = self.models[fine_tuned]
            for required_key in ('repo', 'samplerate', 'sub'):
                if required_key not in model_cfg:
                    error = f'fine_tuned model {fine_tuned} is missing required key {required_key}.'
                    raise ValueError(error)
            sub_dict = model_cfg['sub']
            iso_dir = engine_langs[self.language]
            if iso_dir not in sub_dict:
                error = f'{tts_engine} checkpoint for {self.language} not found.'
                raise KeyError(error)
            self.params['samplerate'] = model_cfg['samplerate']
            self.params['speed'] = float(default_engine_settings[tts_engine].get('speed', 1.0))
            self.params['diffusion_steps'] = int(default_engine_settings[tts_engine].get('diffusion_steps', 10))
            self.model_path = model_cfg['repo']
            enough_vram = self.session['free_vram_gb'] > 4.0
            seed = 0
            self.amp_dtype = self._apply_gpu_policy(enough_vram=enough_vram, seed=seed)
            self.xtts_speakers = self._load_xtts_builtin_list()
            self.device = devices['CUDA']['proc'] if self.session['device'] in [devices['CUDA']['proc'], devices['ROCM']['proc'], devices['JETSON']['proc']] else self.session['device']
            self.engine = self.load_engine()
        except Exception as e:
            error = f'__init__() error: {e}'
            raise ValueError(error)

    def _get_stressify(self)->Any:
        global _stressify
        if _stressify is None:
            from ukrainian_word_stress import Stressifier
            _stressify = Stressifier()
        return _stressify

    def _to_phonemes(self, text:str)->str:
        from unicodedata import normalize
        from ukrainian_word_stress import StressSymbol
        from ipa_uk import ipa
        # '+' after a syllable is the manual stress override the upstream demo uses
        text = text.replace('+', StressSymbol.CombiningAcuteAccent)
        text = normalize('NFKC', text)
        text = text.replace('"', '')
        text = re.sub(r'[᠆‐‑‒–—―⁻₋−⸺⸻]', '-', text)
        text = re.sub(r' - ', ': ', text)
        text = text.strip()
        if not text:
            return ''
        if text[-1] not in '.?!:-':
            text += '.'
        return ipa(self._get_stressify()(text))

    def _get_style(self, voice:str|None, tokens:Any)->Any:
        import torch
        if voice is None:
            voice = default_engine_settings[self.session['tts_engine']]['voice']
        speaker = Path(voice).stem if voice is not None else None
        if speaker in self.style_cache:
            return self.style_cache[speaker]
        builtin = default_engine_settings[self.session['tts_engine']]['voices']
        if speaker in builtin:
            style_file = os.path.join(self.speakers_path, f'{speaker}.pt')
            if not os.path.exists(style_file):
                error = f'_get_style(): missing style embedding {style_file}'
                raise FileNotFoundError(error)
            style = torch.load(style_file, map_location=self.device)
        else:
            # any other wav is cloned zero-shot by the model itself
            msg = f'Extracting StyleTTS2 voice style from {speaker}…'
            print(msg)
            style = self.engine.predict_style_multi(
                voice,
                tokens,
                diffusion_steps=self.params['diffusion_steps']
            )
        style = style.to(self.device)
        self.style_cache[speaker] = style
        return style

    def load_engine(self)->Any:
        try:
            msg = f"Loading TTS {self.tts_key} model, it takes a while, please be patient…"
            print(msg)
            self.cleanup_memory()
            engine = loaded_tts.get(self.tts_key)
            if not engine:
                from styletts2_inference.models import StyleTTS2 as StyleTTS2Model
                self.tts_key = f'{self.session["tts_engine"]}-{self.model_path}'
                engine = loaded_tts.get(self.tts_key)
                if not engine:
                    engine = StyleTTS2Model(hf_path=self.model_path, device=self.device)
                    loaded_tts[self.tts_key] = engine
            if engine:
                msg = f"TTS {self.tts_key} Loaded!"
                print(msg)
                return engine
            error = "load_engine(): engine is None"
            raise RuntimeError(error)
        except Exception as e:
            error = f"load_engine() error: {e}"
            raise RuntimeError(error) from e

    def convert(self, sentence_file:str, sentence:str, **kwargs)->tuple:
        try:
            import torch
            from lib.classes.tts_engines.common.audio import is_audio_data_valid
            if self.engine:
                sentence_parts = self._split_sentence_on_sml(sentence)
                self.params['block_voice'] = kwargs.get('block_voice', self.session['voice'])
                if self.params.get('inline_voice'):
                    self.params['current_voice'] = self.params['inline_voice']
                else:
                    self.params['current_voice'], error = self._set_voice(self.params['block_voice'])
                    if self.params['current_voice'] is None and error is not None:
                        return False, error
                self.speaker = Path(self.params['current_voice']).stem if self.params['current_voice'] is not None else None
                self.audio_segments = []
                for part in sentence_parts:
                    part = part.strip()
                    if not part:
                        continue
                    if SML_TAG_PATTERN.fullmatch(part):
                        success, error = self._convert_sml(part)
                        if success:
                            self.speaker = Path(self.params['current_voice']).stem if self.params['current_voice'] is not None else None
                        else:
                            return False, error
                        continue
                    if not any(c.isalnum() for c in part):
                        continue
                    if part.endswith("'"):
                        part = part[:-1]
                    try:
                        part_ipa = self._to_phonemes(part)
                        if not part_ipa:
                            continue
                        tokens = self.engine.tokenizer.encode(part_ipa)
                        # The tokenizer always builds on CPU; the model lives on
                        # self.device. Handing it a CPU index tensor fails inside
                        # index_select ("index is on cpu, different from other
                        # tensors on cuda:0") the moment a real conversion starts.
                        if torch.is_tensor(tokens):
                            tokens = tokens.to(self.device)
                        style = self._get_style(self.params['current_voice'], tokens)
                        with torch.inference_mode():
                            with torch.autocast(self.device, dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
                                audio_part = self.engine(
                                    tokens,
                                    speed=self.params['speed'],
                                    s_prev=style
                                )
                        if audio_part is not None and len(audio_part) > 0:
                            if torch.is_tensor(audio_part):
                                # On CUDA the model returns half precision, and
                                # soundfile refuses float16 ("dtype must be one of
                                # float32, float64, int16, int32"), so the save
                                # fails at the very last step. Normalize here.
                                audio_part = audio_part.detach().cpu().float()
                            elif str(getattr(audio_part, 'dtype', '')) == 'float16':
                                # numpy path — headers.py doesn't import numpy,
                                # so compare the dtype by name rather than object.
                                audio_part = audio_part.astype('float32')
                            if not is_audio_data_valid(audio_part):
                                error = 'audio_part not valid'
                                return False, error
                            part_tensor = self._tensor_type(audio_part).unsqueeze(0)
                            if part_tensor.numel() == 0:
                                error = 'part_tensor not valid'
                                return False, error
                            self.audio_segments.append(part_tensor)
                        else:
                            error = 'audio_part not valid'
                            return False, error
                    except IndexError as e:
                        error = f'convert() error at {e} segment: {part}'
                        return False, error
                    except Exception as e:
                        return False, self.log_exception(f'{self.__class__.__name__}.convert() part loop', e)
                if self.audio_segments:
                    segment_tensor = torch.cat(self.audio_segments, dim=-1)
                    if not self.audio_save(sentence_file, segment_tensor, self.params['samplerate']):
                        error = f'audio_save() error: cannot save {sentence_file}'
                        return False, error
                    self.audio_segments = []
                    if not os.path.exists(sentence_file):
                        error = f'Cannot create {sentence_file}'
                        return False, error
                return True, None
            else:
                error = f"TTS engine {self.session['tts_engine']} failed to load!"
                return False, error
        except Exception as e:
            self.cleanup_memory()
            self.audio_segments = []
            return False, self.log_exception(f'{self.__class__.__name__}.convert()', e)

    def create_vtt(self, all_sentences:list)->bool:
        if self._build_vtt_file(all_sentences):
            return True
        return False
