from typing import Dict, Any
from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.detection.utils import * 

class DetectionEnv(BaseTextEnv):
    """
    Environment that is used to facilitate a multi-turn interaction between two LLMs. 
    One the "agent" and the other the "target", where the goal is to identify the LLM
    behind the "target".
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        """
        env_config: has keys 
            - 'possible models'
            - 'target_port'
            - 'target_model'
            - 'target_personality'
            - 'detect_model'
            - 'detect_port', 
            - 'ground_truth_model'
            - 'max_turns'
        """
        super().__init__()
        self.detector = Detector(possible_models=env_config['possible_models'], 
                                   port=env_config['detect_port'], 
                                   model=env_config['detect_model'], 
                                   ground_truth_model=env_config['ground_truth_model'])
        
        self.target_agent = TargetAgent(personality_prompt=env_config['target_personality'], 
                                   port=env_config['target_port'],
                                   model=env_config['target_model'])
        self.conversation = []
        self.max_turns = env_config['max_turns']

    def init(self, prompt):
        # No special pre-processing; return prompt and empty metadata
        return prompt, {}

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        new_message = {
            "role": "assistant", 
            "content": action
        }
        self.conversation = self.conversation + [new_message]

        # Per-turn reward: 1.0 if correct, 0.2/max_turns if well-formatted but incorrect, 0.0 otherwise.
        reward = self.detector.calculate_reward(conversation)
        done = self.turns >= self.max_turns

        return BaseTextEnvStepOutput(
            observations=self.conversation,
            reward=reward,
            done=done,
            metadata={},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "steps": self.turns,
        }

    @staticmethod
    def aggregate_metrics(metrics: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not metrics:
            return {}
        n = len(metrics)
        avg_steps = sum(float(m.get("steps", 0)) for m in metrics) / n
        return {"avg_steps": avg_steps}
