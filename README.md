# Cyberpunk City Demo

A real-time rendering demo showcasing ray tracing and volumetric lighting effects in a cyberpunk city environment.

## Features

- **Ray-traced reflections** using Vulkan RTX
- **Volumetric lighting** with light shafts and fog
- **PBR materials** with emissive neon lighting
- **Real-time vehicle movement** on predefined paths
- **Post-processing effects** including bloom and tone mapping

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