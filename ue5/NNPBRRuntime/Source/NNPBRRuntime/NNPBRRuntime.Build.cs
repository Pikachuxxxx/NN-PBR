using UnrealBuildTool;

public class NNPBRRuntime : ModuleRules
{
    public NNPBRRuntime(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine" });
        PrivateDependencyModuleNames.AddRange(new[] { "Json", "Projects", "RenderCore", "RHI" });
        AddEngineThirdPartyPrivateStaticDependencies(Target, "OpenSSL");
    }
}
