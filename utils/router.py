from models.mistral_local import call_mistral
import re

ROUTING_TEMPLATE = """
You are part of a hybrid intelligence system.

There are two models:
- "Scout": Very powerful, handles complex, deep reasoning tasks.
- "Mistral": Lightweight, handles short or basic factual prompts.

Rate how confidently you (Mistral) can answer the following prompt on a scale of 0 to 10.

- 10 = Very confident, you can fully answer it.
- 0 = Not confident at all, you need to escalate to Llama 4 Scout to answer it.

Only return a number between 0 and 10. No explanation.

Prompt:
--------------------
{user_prompt}
--------------------

Only return the response in integer as the response.Do not include any other text or explanation.

Example:
Prompt: "What is the capital of France?"
=> 10
""" 

def confidence(user_prompt: str) -> float:
    routing_prompt = ROUTING_TEMPLATE.format(user_prompt=user_prompt)
    result = call_mistral(routing_prompt)

    match = re.search(r"\b(\d+(?:\.\d+)?)\b", result)
    if match:
        score = float(match.group(1))
        return max(0.0, min(score, 10.0))

    print("[Confidence Routing] Could not parse score:", result)
    return 0.0
    
def route_decider(confidence: float, threshold: float = 6.0) -> str:
    if confidence >= threshold:
        return "edge"
    else:
        return "scout"