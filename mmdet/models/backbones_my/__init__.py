from .clip_model import CLIPModel, TextEncoder
from .clip_prompt_learner import PromptLearner

from .ofa_model import OFA
from .ofa_prompt_learner import OFAPromptLearner
__all__ =[
    'CLIPModel', 'TextEncoder', 'PromptLearner',
    'OFA', 'OFAPromptLearner'
]
