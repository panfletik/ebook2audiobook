from lib.conf_models import TTS_ENGINES, default_engine_settings

models = {
    "internal": {
        "lang": "ukr",
        "repo": "patriotyk/styletts2_ukrainian_multispeaker",
        "voices_repo": "patriotyk/styletts2-ukrainian",
        "sub": {
            "uk": ["patriotyk/styletts2_ukrainian_multispeaker"]
        },
        "voice": default_engine_settings[TTS_ENGINES['STYLETTS2']]['voice'],
        "files": default_engine_settings[TTS_ENGINES['STYLETTS2']]['files'],
        "samplerate": default_engine_settings[TTS_ENGINES['STYLETTS2']]['samplerate']
    }
}
