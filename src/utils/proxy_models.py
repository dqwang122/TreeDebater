"""Model aliases served by the configured Bedrock-compatible proxy."""
PROXY_MODELS = {
    'google.gemma-4-26b-a4b', 'nvidia.nemotron-super-3-120b',
    'claude-opus-4-8', 'gpt-6.1-sol', 'gpt-5.6-sol',
}


def is_proxy_model(model):
    return model.removeprefix('openai/') in PROXY_MODELS


def fixed_sampling(model):
    return model.removeprefix('openai/') in {'claude-opus-4-8', 'gpt-6.1-sol', 'gpt-5.6-sol'}
