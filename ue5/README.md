# NN-PBR for Unreal Engine

The source plugin targets desktop UE 5.6 and 5.8 on hardware with unsigned BC6H sampling. The first engine build and render are pending. Each export supplies one mesh material. The runtime reads the model once when requested. Loading is synchronous, so load during setup and reuse the material instance across meshes that share a model.

![Material loading and shading](runtime.svg)

## First test on UE 5.6

1. Copy the entire `ue5/NNPBRRuntime` folder to `MyGame/Plugins/NNPBRRuntime` in a C++ project. Enable **NN-PBR Runtime** in the project. Close the editor before building.
2. Build the editor target and generate the master material. Substitute the engine path, project path and editor target name in this PowerShell example.

```powershell
$UE = 'D:\Epic\UE_5.6'
$Project = 'D:\Projects\MyGame\MyGame.uproject'
& "$UE\Engine\Build\BatchFiles\Build.bat" MyGameEditor Win64 Development "-Project=$Project" -WaitMutex
if ($LASTEXITCODE -ne 0) { throw 'Editor target failed to build' }
& "$UE\Engine\Binaries\Win64\UnrealEditor-Cmd.exe" $Project -run=NNPBRCreateMaterial -unattended -nop4 -stdout -FullStdOutLogOutput
```

3. Confirm `M_NNPBR.uasset` and `T_NNPBR_Linear.uasset` exist in `Plugins/NNPBRRuntime/Content/Materials`. Keep both with your project. The commandlet skips an existing master.
4. Copy `metadata.json`, `decoder_fp16.bin` and all four `latent_*.bc6.dds` files from `models/hammered-copper-512/export` into `MyGame/Content/NNPBRExports/Copper` using the file manager. Preserve their names and keep them as raw files. In the editor, leave the DDS import dialog closed.
5. On a mesh actor’s **BeginPlay**, call **Load Neural Material** with export directory `NNPBRExports/Copper`. Leave **Parent Material** empty to load the generated master. If the returned instance is valid, pass it to the mesh’s **Set Material** at element 0. Otherwise print the node’s **Error** output.

Use a UV-mapped plane and Buffer Visualization to inspect Base Color, World Normal, Roughness and Metallic. Move the camera away to exercise the uploaded mip chains. Copper uses latent base sizes 128, 64, 32 and 16. UV0 uses clamp addressing. The decoder converts albedo from sRGB to linear and reconstructs a tangent-space normal. The default `NNNormalYSign` is −1 for OpenGL-style normal sources. Set it to +1 on the instance for sources already in Unreal’s convention.

## Packaged build

Include the generated assets and the raw exports in the game. Add these entries to the project's `Config/DefaultGame.ini`, then cook and package a Development build with the test map included.

```ini
[/Script/UnrealEd.ProjectPackagingSettings]
+DirectoriesToAlwaysCook=(Path="/NNPBRRuntime/Materials")
+DirectoriesToAlwaysStageAsUFS=(Path="NNPBRExports")
```

Launch the packaged game on a BC6H-capable device. Verify copper loads from the relative directory. On a second call, try a missing directory and check the node returns an error. A flat default material, shader compile error or missing mip chain is a failed integration test. For feedback, record the engine version, rendering backend and first build or shader error from the project log.

## Standalone checks

From the repo root with CMake, a C++17 compiler and the Vulkan SDK available:

```sh
cmake -S ue5/tests -B build/ue-checks
cmake --build build/ue-checks --config Release
ctest --test-dir build/ue-checks -C Release --output-on-failure
```

The tests parse the shipped DDS and decoder files, validate the compiled shader, and compare its Vulkan output with a CPU reference using 64 synthetic latent vectors per model. Those checks cover the decoder math. The UE build, material graph, texture filtering and cooked file staging need the engine test above.
