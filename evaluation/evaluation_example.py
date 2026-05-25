import random

try:
    from .evaluated_model_example import CausalDiffusionEvaluator
except ImportError:
    from evaluated_model_example import CausalDiffusionEvaluator


def sample_timestep() -> int:
    """Sample an interior DDPM timestep for this legacy illustrative example."""
    return random.randint(1, 999)


# Evaluation script maintained by organizers; participants do not have access to this code
def evaluate_model(evaluator: CausalDiffusionEvaluator, causal_dataset):
    total_causal_score = 0

    for video_normal, video_reversed, prompt in causal_dataset:
        # Randomly or uniformly sample a timestep
        t = sample_timestep()

        # Call the participant's implemented API to get loss
        loss_normal = evaluator.get_denoising_loss(video_normal, prompt, t)
        loss_reversed = evaluator.get_denoising_loss(video_reversed, prompt, t)

        # If the model has causal understanding, the denoising loss for reversed video should be significantly higher than for normal video
        if loss_reversed > loss_normal:
            total_causal_score += 1

    return total_causal_score / len(causal_dataset)
