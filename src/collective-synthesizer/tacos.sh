#!/bin/bash
set -e

# path to the project directory
PROJECT_DIR="$(dirname "$(realpath "$0")")"
# useful directories
BUILD_DIR="$PROJECT_DIR/build"

# Number of build workers. The artifact's top-level driver sets
# CMAKE_BUILD_PARALLEL_LEVEL from GLAIVE_WORKERS.
if command -v nproc >/dev/null 2>&1; then
    DETECTED_THREADS="$(nproc)"
else
    DETECTED_THREADS="$(sysctl -n hw.logicalcpu)"
fi
THREADS="${CMAKE_BUILD_PARALLEL_LEVEL:-${GLAIVE_BUILD_JOBS:-$DETECTED_THREADS}}"
if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: build worker count must be a positive integer" >&2
    exit 1
fi

# functions to compile and run TACOS
configure() {
    local test_flag="OFF"
    local build_type="Release"
    if [[ "${1:-}" == "--with-tests" ]]; then
        test_flag="ON"
        build_type="Debug"
    fi

    echo "[TACOS] Configuring project..."
    if ! cmake -S "$PROJECT_DIR" \
        -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$build_type" \
        -DBUILD_TESTING="$test_flag"; then
        mkdir -p "$BUILD_DIR"
        (
            cd "$BUILD_DIR"
            cmake "$PROJECT_DIR" \
                -DCMAKE_BUILD_TYPE="$build_type" \
                -DBUILD_TESTING="$test_flag"
        )
    fi
}

build() {
    echo "[TACOS] Building project (with $THREADS threads)..."
    cmake --build "$BUILD_DIR" --parallel "$THREADS"
}

# run TACOS
run() {
    # Accept topology, collective, multithread flag, and output directory
    TOPOLOGY_CONFIG=${1:-"input/topology/mesh2d_1.json"}
    COLLECTIVE_CONFIG=${2:-"input/collective/allgather_1.json"}
    MULTITHREAD=${3:-"false"}
    OUTPUT_DIR=${4:-""}
    
    echo "[TACOS] Running with topology: $TOPOLOGY_CONFIG"
    echo "[TACOS] Running with collective: $COLLECTIVE_CONFIG"
    echo "[TACOS] Multithread: $MULTITHREAD"
    
    if [ "$MULTITHREAD" = "true" ]; then
        if [ -z "$OUTPUT_DIR" ]; then
            echo "Error: Output directory must be specified when multithread is enabled"
            exit 1
        fi
        echo "[TACOS] Output directory: $OUTPUT_DIR"
        ./build/bin/tacos "$TOPOLOGY_CONFIG" "$COLLECTIVE_CONFIG" "$MULTITHREAD" "$OUTPUT_DIR"
    else
        ./build/bin/tacos "$TOPOLOGY_CONFIG" "$COLLECTIVE_CONFIG"
    fi
}

test() {
    echo "[TACOS] Running tests (with $THREADS threads)..."
    ctest --test-dir "$BUILD_DIR" --output-on-failure --parallel "$THREADS"
}

clean() {
    echo "[TACOS] Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    rm -rf "$PROJECT_DIR/Testing"
}

usage() {
    local script_name=$(basename "$0")
    echo "Usage:"
    printf "\t%s configure [--with-tests]\t%s\n" "$script_name" "Configure the build system."
    printf "\t%s build\t%s\n" "$script_name" "Build the project."
    printf "\t%s run <topology> <collective> [multithread] [output_dir]\t%s\n" "$script_name" "Run the compiled binary."
    printf "\t%s test\t%s\n" "$script_name" "Run tests."
    printf "\t%s clean\t%s\n" "$script_name" "Clean the build directory."
    printf "\t%s\t%s\n" "$script_name" "Run configure-build-run sequence."
    echo ""
    echo "Examples:"
    printf "\t%s run input/topology/mesh2d_1.json input/collective/allgather_1.json\n" "$script_name"
    printf "\t%s run input/topology/mesh2d_1.json input/collective/allgather_1.json false\n" "$script_name"
    printf "\t%s run input/topology/mesh2d_1.json input/collective/allgather_1.json true results\n" "$script_name"

    exit 1
}

# dispatch commands
case "${1:-}" in
    "")
        configure
        build
        run;;
    configure)
        configure "${2:-}";;
    build)
        build;;
    run)
        # Pass all remaining arguments to run function
        shift
        run "$@"
        ;;
    test)
        test;;
    clean)
        clean;;
    *)
        usage;;
esac

