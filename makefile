# --- Configuration ---

# The submodule directory (used as a fallback)
SUBMODULE_DIR := rlviser_repo

# The name of the executable we want in our project's root directory
INSTALL_TARGET := rlviser

# URL for the pre-built Linux binary
DOWNLOAD_URL := https://github.com/VirxEC/rlviser/releases/download/v0.7.16/rlviser

# The path to the source-built executable (used as a fallback)
SRC_EXECUTABLE := $(SUBMODULE_DIR)/target/release/rlviser

# Detect OS and architecture to decide whether to download or build
# This will result in something like "Linux-x86_64" or "Darwin-arm64"
OS_ARCH := $(shell uname -s)-$(shell uname -m)


# --- Main Targets ---

# The default 'make' command will run this target
.DEFAULT_GOAL := all
all: $(INSTALL_TARGET)

# If the system is Linux x86_64, download the pre-built binary.
ifeq ($(OS_ARCH), Linux-x86_64)
$(INSTALL_TARGET):
	@echo "💻 Detected Linux x86_64. Downloading pre-built binary..."
	@curl --fail --location -o $@ $(DOWNLOAD_URL)
	@chmod +x $@
	@echo "✅ Download complete: $(INSTALL_TARGET)"

# For all other systems (e.g., macOS, Windows, other architectures),
# build from the source in the submodule.
else
$(INSTALL_TARGET): $(SRC_EXECUTABLE)
	@echo "✅ Installing dependency by copying: $(SRC_EXECUTABLE) -> $(INSTALL_TARGET)"
	@cp $< $@

$(SRC_EXECUTABLE):
	@echo "ℹ️ System is not Linux x86_64. Building dependency from source..."
	@cd $(SUBMODULE_DIR) && cargo build --release
endif


# --- Housekeeping ---

# The clean rule removes the installed executable and cleans any build artifacts.
clean:
	@echo "🧹 Cleaning up..."
	@rm -f $(INSTALL_TARGET)
	@# Only try to run 'cargo clean' if the submodule directory exists
	@if [ -d "$(SUBMODULE_DIR)" ]; then \
		echo "Cleaning submodule build artifacts..."; \
		cd $(SUBMODULE_DIR) && cargo clean; \
	fi

# Declare that 'all' and 'clean' are not files.
.PHONY: all clean
