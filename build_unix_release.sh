
## First, cd into the directory containing this script
cd `dirname "$(readlink -f "$0")"`


cmake --build "build" --config Release

