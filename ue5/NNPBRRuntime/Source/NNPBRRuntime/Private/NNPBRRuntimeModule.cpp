#include "Modules/ModuleManager.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"

class FNNPBRRuntimeModule final : public IModuleInterface
{
  public:
    void StartupModule() override
    {
        const auto Plugin = IPluginManager::Get().FindPlugin(TEXT("NNPBRRuntime"));
        check(Plugin);
        AddShaderSourceDirectoryMapping(TEXT("/Plugin/NNPBRRuntime"), Plugin->GetBaseDir() / TEXT("Shaders"));
    }
};
IMPLEMENT_MODULE(FNNPBRRuntimeModule, NNPBRRuntime)
