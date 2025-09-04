# Point to the MoltenVK driver
export VK_ICD_FILENAMES="/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"

# Set Vulkan SDK path
export VULKAN_SDK="/opt/homebrew"

# Add library path 
export DYLD_LIBRARY_PATH="/opt/homebrew/lib:$DYLD_LIBRARY_PATH"

# Set layer path (for validation layers)
export VK_LAYER_PATH="/opt/homebrew/share/vulkan/explicit_layer.d"
