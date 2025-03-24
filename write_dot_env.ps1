# Clear the contents of the .env file
Set-Content -Path .env -Value ""

# Append new values to the .env file
$azureAiChatDeployment = azd env get-value AZURE_AI_CHAT_DEPLOYMENT
$azureAiEndpoint = azd env get-value AZURE_AI_ENDPOINT

Add-Conent -Path .env -Value "API_HOST=azure"
Add-Content -Path .env -Value "AZURE_AI_CHAT_DEPLOYMENT=$azureAiChatDeployment"
Add-Content -Path .env -Value "AZURE_AI_ENDPOINT=$azureAiEndpoint"
