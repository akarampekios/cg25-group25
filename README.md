# Cyberpunk City Demo

A real-time rendering demo showcasing ray tracing and volumetric lighting effects in a cyberpunk city environment.

## Features

- **Ray-traced reflections** using Vulkan RTX
- **Volumetric lighting** with light shafts and fog
- **PBR materials** with emissive neon lighting
- **Real-time vehicle movement** on predefined paths
- **Post-processing effects** including bloom and tone mapping

## Hybrid Rendering Architecture

This project uses a **hybrid rasterization + ray tracing** approach for optimal real-time performance:

- **Rasterization**: Traditional graphics pipeline renders the base scene geometry (vertex + fragment shaders)
- **Inline Ray Queries**: Fragment shader selectively shoots rays using `TraceRayInline()` for specific effects
- **Ray-Traced Shadows**: Shadow rays test occlusion from directional, point, and spot lights
- **Ray-Traced Reflections**: Reflection rays only traced for smooth/metallic surfaces (roughness < 0.3)
- **Acceleration Structures**: BLAS/TLAS enable fast ray-geometry intersection tests (~2-6 rays per pixel)

### What Ray Tracing Actually Computes

**Ray Traced (Hardware Accelerated):**
- **Shadows**: Shadow rays from surface to each light source to test occlusion (1 ray per light: directional + point lights + spot lights in range)
- **Specular Reflections**: Reflection rays for smooth/metallic materials only (1 ray/pixel when applicable)

**NOT Ray Traced (Optimized Alternatives):**
- **Base Geometry**: Rendered via traditional rasterization (much faster than ray tracing every pixel)
- **Diffuse Lighting**: Calculated using PBR equations in fragment shader (no rays needed)
- **Ambient Occlusion**: Pre-baked into textures during asset creation (loaded from GLTF models)
- **Indirect Diffuse**: Approximated by sampling skybox texture based on surface normal

## Performance Optimizations

The project implements several key optimizations for real-time performance:

- **Persistent Mapping**: GPU buffers mapped once at initialization and kept mapped throughout the application lifetime, eliminating map/unmap overhead on every frame
- **Selective Updates**: Only animated objects, lights, and dynamic data are updated each frame - static geometry remains untouched
- **Frames in Flight**: Separate buffer copies per frame-in-flight prevent CPU/GPU synchronization stalls (triple buffering for uniform/instance/light buffers)
- **Early Exits**: TLAS rebuilds and buffer updates skip work entirely when no animated objects have moved
- **Instance Masks**: Per-object ray visibility masks (0x01 = reflective, 0x02 = shadow-casting) allow fine-grained control over which rays hit which geometry
- **Frustum Culling**: CPU-side frustum culling eliminates draw calls for objects outside the camera view before GPU submission
- **Indirect Drawing**: Multi-draw indirect commands batch multiple draw calls into a single GPU submission with minimal CPU overhead

## External Resources

* Soundtracks: 
  * [Cosmic Countdown](https://www.epidemicsound.com/music/tracks/2e5598c9-de18-4c94-a830-b7ef7b7a09fc/) by [Ben Elson](https://www.epidemicsound.com/artists/ben-elson/)
  * [Wanderlust](https://www.epidemicsound.com/music/tracks/a8a07c28-92f0-45db-8332-51b8d876218d/) by [Ben Elson](https://www.epidemicsound.com/artists/ben-elson/)
* Models: [Cyber City](https://assetstore.unity.com/packages/3d/environments/sci-fi/cyberpunk-cyber-city-urp-267305) by [IL.ranch](https://assetstore.unity.com/publishers/11203)

## Development

### Requirements

- Windows 10/11 with latest updates
- Any C++ IDE, preferably supporting CMake Projects, such as CLion, VS Code, or Visual Studio
- Vulkan SDK 1.4+ (slangc must be on PATH)
- NVIDIA RTX 2070 (or better) with updated drivers
- CMake 3.20+

### Quick Start

```powershell
git clone <repository-url>
cd CyberpunkCityDemo
git submodule update --init --recursive
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The compiled binary lives at `build/bin/Release/CyberpunkCityDemo.exe`. Assets and shaders are copied automatically when you build.

To launch a simplified test scene, adjust the `Application.cpp` to point to the `scene_test.glb`.

## Running

Launch from Explorer or the terminal (cw must be the binary folder):

```powershell
build\bin\Debug\CyberpunkCityDemo.exe
build\bin\Release\CyberpunkCityDemo.exe
```

You should see debug output in the console while the renderer runs.

## Next Steps

Please consult the [Wiki](https://github.com/akarampekios/cg25-group25/wiki) for more information.