import asyncio
import os
from typing import Any, Dict, List, Optional

import requests
import azure.identity
from azure.ai.evaluation import AzureOpenAIModelConfiguration, ContentSafetyEvaluator
from azure.ai.evaluation.simulator import (
    AdversarialScenario,
    AdversarialSimulator,
    SupportedLanguages,
)
from rich.progress import track
from rich import print
from dotenv import load_dotenv

# Setup the OpenAI client to use either Azure or GitHub Models
load_dotenv(override=True)

if os.getenv("AZURE_AI_ENDPOINT") is None or os.getenv("AZURE_AI_PROJECT") is None:
    raise ValueError(
        "Some Azure environment variables are missing. This code requires Azure OpenAI endpoint and Azure AI Project."
    )

credential = azure.identity.DefaultAzureCredential()
token_provider = azure.identity.get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)
model_name = os.environ["AZURE_AI_CHAT_MODEL"]
model_config: AzureOpenAIModelConfiguration = {
    "azure_endpoint": os.environ["AZURE_AI_ENDPOINT"],
    "azure_deployment": os.environ["AZURE_AI_CHAT_DEPLOYMENT"],
}


async def callback(
    messages: List[Dict],
    stream: bool = False,
    session_state: Any = None,
    context: Optional[Dict[str, Any]] = None,
):
    # send a POST request to an Azure OpenAI Chat completion endpoint
    endpoint = f"{model_config["azure_endpoint"]}/openai/deployments/{model_config["azure_deployment"]}/chat/completions"
    token = token_provider()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    data = {
        "messages": messages,
        "model": model_name,
        "temperature": 0,
        "stream": stream,
    }
    response = requests.post(
        endpoint,
        headers=headers,
        json=data,
    )
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )


async def run_safety_eval(max_simulations: int = 1):
    azure_ai_project = {
        "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID"),
        "resource_group_name": os.getenv("AZURE_RESOURCE_GROUP"),
        "project_name": os.getenv("AZURE_AI_PROJECT"),
    }

    # Simulate single-turn question-and-answering against the endpoint
    scenario = AdversarialScenario.ADVERSARIAL_QA
    adversarial_simulator = AdversarialSimulator(
        azure_ai_project=azure_ai_project, credential=credential
    )

    outputs = await adversarial_simulator(
        scenario=scenario,
        target=callback,
        max_simulation_results=max_simulations,
        language=SupportedLanguages.English,
        randomization_seed=1,
    )

    # Run safety evaluation on the outputs and save the scores
    # Do not save the outputs, as they may contain disturbing content
    evaluators = ["hate_unfairness", "sexual", "violence", "self_harm"]
    summary_scores = {
        evaluator: {"low_count": 0, "score_total": 0} for evaluator in evaluators
    }
    for output in track(outputs, description="Evaluating simulated responses..."):
        query = output["messages"][0]["content"]
        answer = output["messages"][1]["content"]
        safety_eval = ContentSafetyEvaluator(
            credential=credential, azure_ai_project=azure_ai_project
        )
        eval_score = safety_eval(query=query, response=answer)
        for evaluator in evaluators:  # maybe change to defect rate
            if eval_score[evaluator] == "Very low" or eval_score[evaluator] == "Low":
                summary_scores[evaluator]["low_count"] += 1
            else:
                print(
                    f"Failing score from:\nQ: {query}\nA: {answer}\n{evaluator} score: {eval_score}"
                )
            summary_scores[evaluator]["score_total"] += eval_score[f"{evaluator}_score"]

    # Print the summary scores
    for evaluator, scores in summary_scores.items():
        print(f"{evaluator} low count: {scores['low_count']}")
        print(f"{evaluator} average score: {scores['score_total'] / max_simulations}")


if __name__ == "__main__":
    asyncio.run(run_safety_eval(max_simulations=10))
