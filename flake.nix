{
  description = "Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        libPath = with pkgs; lib.makeLibraryPath [
          libGL
          libxkbcommon
          # wayland
          vulkan-tools
          vulkan-loader
          # vulkan-caps-viewer
          vulkan-headers
          # vulkan-extension-layer
          vulkan-validation-layers
          vulkan-utility-libraries
          # vulkan-tools-lunarg
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cargo
            rustc
            rust-analyzer
          ];

          packages = with pkgs; [
            # valgrind
            gdb
          ];

          RUST_LOG = "debug";
          RUST_SRC_PATH = "${pkgs.rust.packages.stable.rustPlatform.rustLibSrc}";
          LD_LIBRARY_PATH = libPath;
          VULKAN_SDK = libPath;
          VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };
      }
    );
}
