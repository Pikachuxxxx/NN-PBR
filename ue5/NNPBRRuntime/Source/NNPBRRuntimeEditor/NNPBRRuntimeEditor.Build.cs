using UnrealBuildTool;

public class NNPBRRuntimeEditor : ModuleRules
{
    public NNPBRRuntimeEditor(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
        PublicDependencyModuleNames.AddRange(new[] { "Core", "CoreUObject", "Engine" });
        PrivateDependencyModuleNames.AddRange(new[] { "UnrealEd", "AssetRegistry", "MaterialEditor", "NNPBRRuntime", "ToolMenus", "ContentBrowser", "DesktopPlatform", "PropertyEditor", "Slate", "SlateCore", "InputCore" });
    }
}
