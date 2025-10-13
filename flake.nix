{
  description = "Rust/Vulkan development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        isLinux = pkgs.stdenv.isLinux;
        isDarwin = pkgs.stdenv.isDarwin;

        # Graphics/runtime libs: add Wayland only on Linux.
        gfxLibs = with pkgs; [
          libGL
          libxkbcommon
          vulkan-loader
          vulkan-validation-layers
          vulkan-utility-libraries
        ] ++ lib.optionals isLinux [
          wayland
        ];

        libPath = pkgs.lib.makeLibraryPath gfxLibs;

        linuxEnv = lib.optionalAttrs isLinux {
          LD_LIBRARY_PATH = libPath;
        };

        darwinEnv = lib.optionalAttrs isDarwin {
          # Homebrew path is common on Apple Silicon; harmless if absent.
          DYLD_LIBRARY_PATH = "/opt/homebrew/lib:${libPath}";
        };
      in
      {
        devShells.default = pkgs.mkShell ({
          buildInputs = with pkgs; [
            cargo
            rustc
            rust-analyzer
          ];

          packages = with pkgs; [
            gdb
          ];

          # Common env
          RUST_LOG = "debug";
          RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
          VULKAN_SDK = "${pkgs.vulkan-headers}";
          VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        } // linuxEnv // darwinEnv);
      }
    );
}
