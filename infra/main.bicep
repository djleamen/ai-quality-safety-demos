targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Location for the OpenAI and Azure AI Project resources')
// https://learn.microsoft.com/azure/ai-foundry/how-to/develop/evaluate-sdk#region-support
@allowed([
  'eastus2'
  'francecentral'
  'swedencentral'
  'switzerlandwest'
])
@metadata({
  azd: {
    type: 'location'
  }
})
param location string

@description('Name of the GPT model to deploy')
param gptModelName string = 'gpt-4o'

@description('Version of the GPT model to deploy')
// See version availability in this table:
// https://learn.microsoft.com/azure/ai-services/openai/concepts/models?tabs=python-secure%2Cglobal-standard%2Cstandard-chat-completions#models-by-deployment-type
param gptModelVersion string = '2024-11-20'

@description('Name of the model deployment (can be different from the model name)')
param gptDeploymentName string = 'gpt-4o'

@description('Capacity of the GPT deployment')
// You can increase this, but capacity is limited per model/region, so you will get errors if you go over
// https://learn.microsoft.com/en-us/azure/ai-services/openai/quotas-limits
param gptDeploymentCapacity int = 30

@description('Id of the user or app to assign application roles')
param principalId string = ''

@description('Non-empty if the deployment is running on GitHub Actions')
param runningOnGitHub string = ''

var principalType = empty(runningOnGitHub) ? 'User' : 'ServicePrincipal'

var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var prefix = take(replace(toLower('${environmentName}${resourceToken}'), '-', ''), 15)
var tags = { 'azd-env-name': environmentName }

// Organize resources in a resource group
resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
    name: '${prefix}-rg'
    location: location
    tags: tags
}

var openAiServiceName = '${prefix}-openai'
module openAi 'br/public:avm/res/cognitive-services/account:0.7.1' = {
  name: 'openai'
  scope: resourceGroup
  params: {
    name: openAiServiceName
    location: location
    tags: tags
    kind: 'OpenAI'
    sku: 'S0'
    customSubDomainName: openAiServiceName
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
    deployments: [
      {
        name: gptDeploymentName
        model: {
          format: 'OpenAI'
          name: gptModelName
          version: gptModelVersion
        }
        sku: {
          name: 'GlobalStandard'
          capacity: gptDeploymentCapacity
        }
      }
    ]
    roleAssignments: [
      {
        principalId: principalId
        roleDefinitionIdOrName: 'Cognitive Services OpenAI User'
        principalType: principalType
      }
    ]
  }
}


module storage 'br/public:avm/res/storage/storage-account:0.9.1' = {
  name: 'storage'
  scope: resourceGroup
  params: {
    name: '${prefix}storage'
    location: location
    tags: tags
    kind: 'StorageV2'
    skuName: 'Standard_LRS'
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
  }
}

module ai 'core/ai/ai-environment.bicep' = {
  name: 'ai'
  scope: resourceGroup
  params: {
    location: location
    tags: tags
    hubName: 'aihub-${resourceToken}'
    projectName: 'aiproj-${resourceToken}'
    storageAccountId: storage.outputs.resourceId
    applicationInsightsId: ''
  }
}

output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_RESOURCE_GROUP string = resourceGroup.name

// Specific to Azure OpenAI
output AZURE_AI_SERVICE string = openAi.outputs.name
output AZURE_AI_ENDPOINT string = openAi.outputs.endpoint
output AZURE_AI_CHAT_MODEL string = gptModelName
output AZURE_AI_CHAT_DEPLOYMENT string = gptDeploymentName
output AZURE_AI_PROJECT string = ai.outputs.projectName
