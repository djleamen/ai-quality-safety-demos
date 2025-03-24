#!/bin/bash

# Clear the contents of the .env file
> .env

# Append new values to the .env file
echo "API_HOST=azure" >> .env
echo "AZURE_AI_CHAT_DEPLOYMENT=$(azd env get-value AZURE_AI_CHAT_DEPLOYMENT)" >> .env
echo "AZURE_AI_ENDPOINT=$(azd env get-value AZURE_AI_ENDPOINT)" >> .env
