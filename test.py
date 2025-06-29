from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.credentials import Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Config
API_KEY = "UV_-V1ZuzrdYv6TVGBOp6cMoM7WTHewesFUKHjHbciQg"
PROJECT_ID = "2746f6e3-e042-44c5-a968-d1e4ae1f42f0"
URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/granite-3-8b-instruct"

# Auth
creds = Credentials(api_key=API_KEY, url=URL)

# Model interface
model = ModelInference(
    model_id=MODEL_ID,
    credentials=creds,
    project_id=PROJECT_ID
)

# Generation parameters
generation_params = {
    GenParams.MAX_NEW_TOKENS: 200,
    GenParams.TEMPERATURE: 0.7
}

# Generate
response = model.generate(
    prompt="Summarize the risk scoring in automated loan approval.",
    params=generation_params
)

print(response["results"][0]["generated_text"])
